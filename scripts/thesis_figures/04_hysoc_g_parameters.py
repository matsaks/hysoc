"""HYSOC-G parameter sensitivity figure.

2x3 grid showing how each key parameter (D, T, C, dp_eps) changes the
compressed representation of the same synthetic trajectory.
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
from engines.squish_dp import HybridSquishDPCompressor
from engines.stop_compressor import StopCompressor
from core.point import Point
from core.segment import Stop, Move
from core.squish_dp_config import HybridSquishDPConfig

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

C_RAW   = "#aaaaaa"
C_MOVE  = "#2166ac"
C_STOP  = "#d6604d"
C_KP    = "#2166ac"

BASE_LAT = 40.750000
BASE_LON = -74.000000
M_PER_DEG_LAT = 111319.5
M_PER_DEG_LON = 111319.5 * math.cos(math.radians(BASE_LAT))


def m_to_latlon(dx: float, dy: float) -> tuple[float, float]:
    return BASE_LAT + dy / M_PER_DEG_LAT, BASE_LON + dx / M_PER_DEG_LON


def make_point(dx: float, dy: float, t: datetime, noise_m: float = 0.0) -> Point:
    rng_dx = np.random.normal(0, noise_m) if noise_m else 0.0
    rng_dy = np.random.normal(0, noise_m) if noise_m else 0.0
    lat, lon = m_to_latlon(dx + rng_dx, dy + rng_dy)
    return Point(lat=lat, lon=lon, timestamp=t, obj_id="syn")


def build_trajectory(seed: int = 42) -> list[Point]:
    """Move (22 pts) -> Stop 60 s at (0, 60) -> Move (22 pts).

    Stop centre is 30 m above the move-1 peak so D=15 m leaves a clean gap
    while D=35 m visibly absorbs the last few move-1 points. Stop duration is
    60 s so T=30 s qualifies but T=80 s does not.
    """
    np.random.seed(seed)
    t0 = datetime(2024, 1, 1, 9, 0, 0)
    dt = timedelta(seconds=5)
    pts: list[Point] = []

    n1 = 22
    xs1 = np.linspace(-90, 0, n1)
    ys1 = -60 + 90 * (1 - ((xs1 + 90) / 90 - 1) ** 2)
    for i in range(n1):
        pts.append(make_point(xs1[i], ys1[i], t0 + i * dt, noise_m=2.5))

    t_stop = t0 + n1 * dt
    for i in range(13):
        pts.append(make_point(0.0, 60.0, t_stop + i * dt, noise_m=4.0))

    t_m2 = t_stop + 13 * dt
    n3 = 22
    xs3 = np.linspace(0, 90, n3)
    ys3 = 60 - 90 * ((xs3 / 90) ** 2)
    for i in range(n3):
        pts.append(make_point(xs3[i], ys3[i], t_m2 + i * dt, noise_m=2.5))

    return pts


def point_xy(p: Point) -> tuple[float, float]:
    return (p.lon - BASE_LON) * M_PER_DEG_LON, (p.lat - BASE_LAT) * M_PER_DEG_LAT


def run(pts: list[Point], D: float, T: float, C: int, dp_eps: float) -> dict:
    """Segment and compress with given parameters."""
    seg = STEPSegmenter(max_eps=D, min_duration_seconds=T)
    segments = seg.process(pts)

    sc = StopCompressor()
    mc = HybridSquishDPCompressor(HybridSquishDPConfig(capacity=C, dp_epsilon_meters=dp_eps))

    result: dict = {"stops": [], "moves_raw": [], "moves_kp": []}
    for s in segments:
        if isinstance(s, Stop):
            cs = sc.compress(s.points)
            result["stops"].append({"raw": s.points, "repr": cs.centroid,
                                    "t0": cs.start_time, "t1": cs.end_time})
        elif isinstance(s, Move):
            kps = mc.compress(s.points)
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


def draw_result(ax: plt.Axes, data: dict) -> None:
    for mv_pts, kps in zip(data["moves_raw"], data["moves_kp"]):
        xy_raw = [point_xy(p) for p in mv_pts]
        ax.scatter([p[0] for p in xy_raw], [p[1] for p in xy_raw],
                   s=6, color=C_RAW, zorder=2, linewidths=0, alpha=0.5)
        xy_kp = [point_xy(p) for p in kps]
        if len(xy_kp) >= 2:
            ax.plot([p[0] for p in xy_kp], [p[1] for p in xy_kp],
                    color=C_MOVE, lw=1.4, zorder=3)
        ax.scatter([p[0] for p in xy_kp], [p[1] for p in xy_kp],
                   s=28, color=C_KP, marker="^", zorder=5, linewidths=0)

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
        plt.Line2D([0], [0], color=C_MOVE, lw=1.4,
                   marker="^", markersize=6, label="MOVE keypoints"),
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

    ref = run(pts, D=15, T=30, C=8, dp_eps=5)
    draw_result(axes[1], ref)
    style_ax(axes[1], xlim, ylim, "(b) Reference HYSOC-G",
             "$D$=15 m · $T$=30 s\n$C$=8 · $\\varepsilon_{dp}$=5 m")

    d_hi = run(pts, D=35, T=30, C=8, dp_eps=5)
    draw_result(axes[2], d_hi)
    style_ax(axes[2], xlim, ylim, "(c) $D$ increases",
             "$D$=35 m\nlarger stop neighbourhood")

    t_hi = run(pts, D=15, T=80, C=8, dp_eps=5)
    draw_result(axes[3], t_hi)
    style_ax(axes[3], xlim, ylim, "(d) $T$ increases",
             "$T$=80 s\nstop (60 s) too short")

    c_lo = run(pts, D=15, T=30, C=4, dp_eps=5)
    draw_result(axes[4], c_lo)
    style_ax(axes[4], xlim, ylim, "(e) $C$ decreases",
             "$C$=4\nfewer SQUISH keypoints")

    dp_hi = run(pts, D=15, T=30, C=8, dp_eps=30)
    draw_result(axes[5], dp_hi)
    style_ax(axes[5], xlim, ylim, "(f) $\\varepsilon_{dp}$ increases",
             "$\\varepsilon_{dp}$=30 m\nstronger DP simplification")

    fig.legend(handles=shared_legend_handles(), loc="lower center", ncol=3,
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
