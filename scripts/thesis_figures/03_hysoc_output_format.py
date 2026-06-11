"""HYSOC output format explainer figure.

Three panels showing the progression from raw GPS points through behavioural
segmentation to the compressed HYSOC representation.
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

C_RAW = "#aaaaaa"
C_MOVE = "#2166ac"
C_STOP_RAW = "#d6604d"
C_STOP_COMP = "#d6604d"
C_KEYPOINT = "#2166ac"
C_STOP_REPR = "#d6604d"

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


def build_synthetic_trajectory(seed: int = 42) -> list[Point]:
    """Build a synthetic trajectory: move → stop → move."""
    np.random.seed(seed)
    t0 = datetime(2024, 1, 1, 9, 0, 0)
    dt = timedelta(seconds=5)
    points: list[Point] = []

    n1 = 22
    xs1 = np.linspace(-90, 0, n1)
    ys1 = -60 + 90 * (1 - ((xs1 + 90) / 90 - 1) ** 2)
    for i in range(n1):
        points.append(make_point(xs1[i], ys1[i], t0 + i * dt, noise_m=2.0))

    t_stop_start = t0 + n1 * dt
    for i in range(13):
        points.append(make_point(0.0, 60.0, t_stop_start + i * dt, noise_m=4.0))

    t_move2_start = t_stop_start + 13 * dt
    n3 = 22
    xs3 = np.linspace(0, 90, n3)
    ys3 = 60 - 90 * ((xs3 / 90) ** 2)
    for i in range(n3):
        points.append(make_point(xs3[i], ys3[i], t_move2_start + i * dt, noise_m=2.0))

    return points


def point_to_xy(p: Point) -> tuple[float, float]:
    """Convert Point back to local metre coordinates for plotting."""
    dx = (p.lon - BASE_LON) * M_PER_DEG_LON
    dy = (p.lat - BASE_LAT) * M_PER_DEG_LAT
    return dx, dy


def compress_trajectory(points: list[Point]) -> dict:
    """Run STEP segmentation and HYSOC-G compression. Return plot-ready data."""
    segmenter = STEPSegmenter(max_eps=15.0, min_duration_seconds=30.0)
    segments = segmenter.process(points)

    stop_compressor = StopCompressor()
    move_compressor = HybridSquishDPCompressor(
        HybridSquishDPConfig(capacity=8, dp_epsilon_meters=5.0)
    )

    result = {
        "stop_raw_points": [],
        "move_raw_points": [],
        "stop_repr": None,
        "stop_time_range": None,
        "move_keypoints": [],
    }

    for seg in segments:
        if isinstance(seg, Stop):
            result["stop_raw_points"] = seg.points
            cs = stop_compressor.compress(seg.points)
            result["stop_repr"] = cs.centroid
            result["stop_time_range"] = (cs.start_time, cs.end_time)
        elif isinstance(seg, Move):
            result["move_raw_points"].append(seg.points)
            kps = move_compressor.compress(seg.points)
            result["move_keypoints"].append(kps)

    return result


STYLE = {
    "raw_dot": dict(s=14, color=C_RAW, zorder=2, linewidths=0),
    "stop_dot": dict(s=18, color=C_STOP_RAW, zorder=3, linewidths=0),
    "move_line": dict(color=C_MOVE, linewidth=1.2, alpha=0.5, zorder=2),
    "move_dot": dict(s=14, color=C_MOVE, zorder=3, linewidths=0),
    "keypoint_dot": dict(s=36, color=C_KEYPOINT, marker="*", zorder=5, linewidths=0),
    "stop_repr_dot": dict(s=120, color=C_STOP_REPR, marker="s", zorder=5,
                          linewidths=0.8, edgecolors="white"),
}


def _ax_limits(all_xy: list[tuple[float, float]], pad: float = 15.0):
    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def panel_raw(ax: plt.Axes, points: list[Point]) -> None:
    xy = [point_to_xy(p) for p in points]
    ax.scatter([p[0] for p in xy], [p[1] for p in xy], **STYLE["raw_dot"])
    ax.set_title("(a) Raw GPS stream", fontsize=9, pad=4)


def panel_segmented(ax: plt.Axes, data: dict, points: list[Point]) -> None:
    for move_pts in data["move_raw_points"]:
        xy = [point_to_xy(p) for p in move_pts]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], **STYLE["move_line"])
        ax.scatter([p[0] for p in xy], [p[1] for p in xy], **STYLE["move_dot"])
    stop_xy = [point_to_xy(p) for p in data["stop_raw_points"]]
    ax.scatter([p[0] for p in stop_xy], [p[1] for p in stop_xy], **STYLE["stop_dot"])
    ax.set_title("(b) STEP segmentation", fontsize=9, pad=4)


def panel_compressed(ax: plt.Axes, data: dict, points: list[Point]) -> None:
    for kps in data["move_keypoints"]:
        xy = [point_to_xy(p) for p in kps]
        ax.plot([p[0] for p in xy], [p[1] for p in xy],
                color=C_KEYPOINT, linewidth=1.5, zorder=3)
        ax.scatter([p[0] for p in xy], [p[1] for p in xy], **STYLE["keypoint_dot"])

    if data["stop_repr"]:
        sx, sy = point_to_xy(data["stop_repr"])
        ax.scatter([sx], [sy], **STYLE["stop_repr_dot"])
        t0, t1 = data["stop_time_range"]
        label = f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
        ax.annotate(label, (sx, sy), xytext=(sx - 50, sy + 8),
                    fontsize=8, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
                    color="#555555")

    ax.set_title("(c) Compressed HYSOC output", fontsize=9, pad=4)


def main() -> None:
    points = build_synthetic_trajectory()
    data = compress_trajectory(points)

    all_xy = [point_to_xy(p) for p in points]
    xlim = (_ax_limits(all_xy)[0], _ax_limits(all_xy)[1])
    ylim = (_ax_limits(all_xy)[2], _ax_limits(all_xy)[3])

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2))
    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    panel_raw(axes[0], points)
    panel_segmented(axes[1], data, points)
    panel_compressed(axes[2], data, points)

    leg_handles = [
        mpatches.Patch(color=C_MOVE, label="MOVE points"),
        mpatches.Patch(color=C_STOP_RAW, label="STOP cluster"),
        plt.Line2D([0], [0], color=C_KEYPOINT, lw=1.5,
                   marker="*", markersize=7, label="MOVE keypoints"),
        plt.Line2D([0], [0], color="none", marker="s", markersize=7,
                   markerfacecolor=C_STOP_REPR, label="STOP representative"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
