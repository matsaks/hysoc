"""HYSOC-N parameter sensitivity figure.

2x3 grid showing how D, T, and the speed threshold gamma change the compressed
representation for HYSOC-N's speed-based move compressor (TRACE Stage 1). The
referential k parameter affects only byte cost, so it is not visualised here.
"""

# ruff: noqa: E402

import math
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from engines.step import STEPSegmenter
from engines.stop_compressor import StopCompressor
from core.point import Point
from core.segment import Stop, Move

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

C_RAW  = "#aaaaaa"
C_MOVE = "#1a7837"
C_STOP = "#d6604d"
C_KP   = "#1a7837"

BASE_LAT = 40.750000
BASE_LON = -74.000000
M_PER_DEG_LAT = 111319.5
M_PER_DEG_LON = 111319.5 * math.cos(math.radians(BASE_LAT))


def m_to_latlon(dx: float, dy: float) -> tuple[float, float]:
    return BASE_LAT + dy / M_PER_DEG_LAT, BASE_LON + dx / M_PER_DEG_LON


def make_point(dx: float, dy: float, t: datetime, noise_m: float = 0.0,
               road_id: int = 0) -> Point:
    rng_dx = np.random.normal(0, noise_m) if noise_m else 0.0
    rng_dy = np.random.normal(0, noise_m) if noise_m else 0.0
    lat, lon = m_to_latlon(dx + rng_dx, dy + rng_dy)
    return Point(lat=lat, lon=lon, timestamp=t, obj_id="syn", road_id=road_id)


EARTH_RADIUS_M = 6371000.0


def _haversine(p1: Point, p2: Point) -> float:
    lat1, lat2 = math.radians(p1.lat), math.radians(p2.lat)
    dlat = lat2 - lat1
    dlon = math.radians(p2.lon - p1.lon)
    x = dlon * math.cos((lat1 + lat2) / 2.0)
    return EARTH_RADIUS_M * math.sqrt(x * x + dlat * dlat)


def _arc_place(x_dense: np.ndarray, y_dense: np.ndarray,
               step_pattern: list[float], noise_m: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Resample (x_dense, y_dense) at non-uniform arc-length positions.

    Speed is encoded in the step sizes (large step = fast, small step = slow),
    while dt remains uniform so STEP sees no spurious dwell clusters.
    The step pattern is rescaled to fit 97% of the path arc length.
    """
    dx = np.diff(x_dense)
    dy = np.diff(y_dense)
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total = arc[-1]
    scale = total * 0.97 / sum(step_pattern)
    targets = np.cumsum([0.0] + [s * scale for s in step_pattern])
    xs = np.interp(targets, arc, x_dense) + rng.normal(0, noise_m, len(targets))
    ys = np.interp(targets, arc, y_dense) + rng.normal(0, noise_m, len(targets))
    return xs, ys


def build_trajectory(seed: int = 42) -> list[Point]:
    """Move (22 pts, arc-based speed variation, 2 roads) -> Stop 60 s -> Move (22 pts).

    Speed variation uses non-uniform spatial spacing with uniform dt_move=1 s:
    fast step ~8.2 m/s, slow step ~2.1 m/s, transition change ~6.1 m/s (detects
    at gamma=5, not gamma=10). Stop centre at (0, 60), 30 m above the parabola
    peak and outside D=15 m, lasting 60 s so T=30 s qualifies but T=80 s does not.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, 9, 0, 0)
    dt_move = timedelta(seconds=1)
    dt_stop = timedelta(seconds=5)
    pts: list[Point] = []

    steps_fsf = [20] * 7 + [5] * 7 + [20] * 7   # fast / slow / fast

    x_dense1 = np.linspace(-90, 0, 2000)
    y_dense1 = -60 + 90 * (1 - ((x_dense1 + 90) / 90 - 1) ** 2)
    xs1, ys1 = _arc_place(x_dense1, y_dense1, steps_fsf, noise_m=0.5, rng=rng)
    for i, (x, y) in enumerate(zip(xs1, ys1)):
        rid = 1 if i < len(xs1) // 2 else 2
        pts.append(make_point(x, y, t0 + i * dt_move, road_id=rid))

    t_stop = t0 + len(xs1) * dt_move
    for i in range(13):
        pts.append(make_point(0.0, 60.0, t_stop + i * dt_stop, noise_m=2.0, road_id=3))
    t_m2 = t_stop + 13 * dt_stop

    x_dense2 = np.linspace(0, 90, 2000)
    y_dense2 = 60 - 90 * ((x_dense2 / 90) ** 2)
    xs2, ys2 = _arc_place(x_dense2, y_dense2, steps_fsf, noise_m=0.5, rng=rng)
    for i, (x, y) in enumerate(zip(xs2, ys2)):
        rid = 4 if i < len(xs2) // 2 else 5
        pts.append(make_point(x, y, t_m2 + i * dt_move, road_id=rid))

    return pts


def point_xy(p: Point) -> tuple[float, float]:
    return (p.lon - BASE_LON) * M_PER_DEG_LON, (p.lat - BASE_LAT) * M_PER_DEG_LAT


def speed_based_repr(points: list[Point], gamma: float) -> list[Point]:
    """Return keypoints retained by the γ-threshold speed filter."""
    if not points:
        return []
    retained = [points[0]]
    current_road = points[0].road_id
    last_speed = 0.0

    for i in range(1, len(points)):
        p, prev = points[i], points[i - 1]
        dist = _haversine(prev, p)
        dt = (p.timestamp - prev.timestamp).total_seconds()
        speed = dist / dt if dt > 0 else last_speed

        if p.road_id != current_road:
            current_road = p.road_id
            last_speed = speed
            retained.append(p)
            continue

        if abs(speed - last_speed) > gamma:
            last_speed = speed
            retained.append(p)

    return retained


def run(pts: list[Point], D: float, T: float, gamma: float) -> dict:
    seg = STEPSegmenter(max_eps=D, min_duration_seconds=T)
    segments = seg.process(pts)

    sc = StopCompressor()
    result: dict = {"stops": [], "moves_raw": [], "moves_kp": []}
    for s in segments:
        if isinstance(s, Stop):
            cs = sc.compress(s.points)
            result["stops"].append({"raw": s.points, "repr": cs.centroid,
                                    "t0": cs.start_time, "t1": cs.end_time})
        elif isinstance(s, Move):
            kps = speed_based_repr(s.points, gamma)
            result["moves_raw"].append(s.points)
            result["moves_kp"].append(kps)
    return result


def _ax_limits(pts: list[Point], pad: float = 15.0):
    xs = [(p.lon - BASE_LON) * M_PER_DEG_LON for p in pts]
    ys = [(p.lat - BASE_LAT) * M_PER_DEG_LAT for p in pts]
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_raw(ax: plt.Axes, pts: list[Point]) -> None:
    xy = [point_xy(p) for p in pts]
    ax.scatter([p[0] for p in xy], [p[1] for p in xy],
               s=9, color=C_RAW, zorder=2, linewidths=0)


_ROAD_PALETTE = ["#2166ac", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"]


def _road_color(road_id) -> str:
    return _ROAD_PALETTE[int(road_id) % len(_ROAD_PALETTE)]


def draw_result(ax: plt.Axes, data: dict) -> None:
    """Draw HYSOC-N compressed output.

    Move segments are rendered as road-geometry arcs (coloured by road ID)
    because HYSOC-N reconstructs the path from the road network, not from
    straight lines between sparse keypoints.
    """
    for mv_pts, kps in zip(data["moves_raw"], data["moves_kp"]):
        current_rid = mv_pts[0].road_id
        seg_x: list[float] = []
        seg_y: list[float] = []
        for p in mv_pts:
            if p.road_id != current_rid:
                ax.plot(seg_x, seg_y, color=_road_color(current_rid),
                        lw=1.8, zorder=3, solid_capstyle="round")
                seg_x, seg_y = [], []
                current_rid = p.road_id
            x, y = point_xy(p)
            seg_x.append(x)
            seg_y.append(y)
        if seg_x:
            ax.plot(seg_x, seg_y, color=_road_color(current_rid),
                    lw=1.8, zorder=3, solid_capstyle="round")

        xy_kp = [point_xy(p) for p in kps]
        ax.scatter([p[0] for p in xy_kp], [p[1] for p in xy_kp],
                   s=36, color=C_KP, marker="^", zorder=5, linewidths=0.6,
                   edgecolors="white")

    for stop in data["stops"]:
        raw_xy = [point_xy(p) for p in stop["raw"]]
        ax.scatter([p[0] for p in raw_xy], [p[1] for p in raw_xy],
                   s=9, color=C_STOP, zorder=3, linewidths=0, alpha=0.4)
        rx, ry = point_xy(stop["repr"])
        ax.scatter([rx], [ry], s=90, color=C_STOP, marker="s",
                   zorder=6, linewidths=0.8, edgecolors="white")


def count_label(data: dict) -> str:
    n_kp = sum(len(k) for k in data["moves_kp"])
    n_stop = len(data["stops"])
    return f"{n_stop} stop{'s' if n_stop != 1 else ''}, {n_kp} move kpts"


def style_ax(ax: plt.Axes, xlim, ylim, title: str, subtitle: str = "") -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, fontsize=9, pad=4, linespacing=1.4,
                 multialignment="center")


def shared_legend_handles() -> list:
    return [
        plt.Line2D([0], [0], color=_ROAD_PALETTE[0], lw=2.0,
                   label="Road segment (E-seq)"),
        plt.Line2D([0], [0], color="none", marker="^", markersize=7,
                   markerfacecolor=C_KP, label="Retained keypoint"),
        plt.Line2D([0], [0], color="none", marker="s", markersize=7,
                   markerfacecolor=C_STOP, label="STOP representative"),
        mpatches.Patch(color=C_RAW, alpha=0.5, label="Raw points"),
    ]


def main() -> None:
    pts = build_trajectory()
    xlim = _ax_limits(pts)[:2]
    ylim = _ax_limits(pts)[2:]

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.2))
    axes = axes.flatten()

    for ax in axes:
        style_ax(ax, xlim, ylim, "")

    draw_raw(axes[0], pts)
    style_ax(axes[0], xlim, ylim, "(a) Raw trajectory")

    ref = run(pts, D=15, T=30, gamma=5)
    draw_result(axes[1], ref)
    style_ax(axes[1], xlim, ylim, "(b) Reference HYSOC-N",
             "$D$=15 m · $T$=30 s\n$\\gamma$=5 m/s")

    d_hi = run(pts, D=35, T=30, gamma=5)
    draw_result(axes[2], d_hi)
    style_ax(axes[2], xlim, ylim, "(c) $D$ increases",
             "$D$=35 m\nlarger stop neighbourhood")

    t_hi = run(pts, D=15, T=80, gamma=5)
    draw_result(axes[3], t_hi)
    style_ax(axes[3], xlim, ylim, "(d) $T$ increases",
             "$T$=80 s\nstop (60 s) too short")

    g_hi = run(pts, D=15, T=30, gamma=10)
    draw_result(axes[4], g_hi)
    style_ax(axes[4], xlim, ylim, "(e) $\\gamma$ increases",
             "$\\gamma$=10 m/s\nonly road transitions retained")

    g_lo = run(pts, D=15, T=30, gamma=1)
    draw_result(axes[5], g_lo)
    style_ax(axes[5], xlim, ylim, "(f) $\\gamma$ decreases",
             "$\\gamma$=1 m/s\nnear-lossless retention")

    fig.legend(handles=shared_legend_handles(), loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.97, bottom=0.10,
                        left=0.02, right=0.98)

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
