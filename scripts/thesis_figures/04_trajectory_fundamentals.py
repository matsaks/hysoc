"""Render a single NYC trajectory illustrating trajectory data fundamentals.

Left panel shows the full trajectory on a basemap; the right panel zooms into
a short sub-segment so discrete GPS fixes are distinguishable from the
interpolated polyline reconstruction.
"""

import csv
import logging
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(project_root, "data", "raw", "NYC_Evaluation_1000")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

TRAJECTORY_FILE = "4209291.csv"
ZOOM_HALF_WINDOW = 30
ZOOM_AUTO_SELECT = True
ZOOM_CENTER_FRACTION = 0.50

COLOR_LINE = "#0072B2"
COLOR_DOT = "#D55E00"
COLOR_START = "#009E73"
COLOR_END = "#CC79A7"
COLOR_FRAME = "#222222"

FIG_SIZE = (13, 6.2)
DPI = 400


def load_trajectory(filepath: str) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue
            coords.append((lon, lat))
    return coords


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        from shapely.geometry import Point
    except ImportError as e:
        logger.error(f"Required GIS dependency missing: {e}")
        sys.exit(1)

    path = os.path.join(DATA_DIR, TRAJECTORY_FILE)
    coords = load_trajectory(path)
    if len(coords) < 2 * ZOOM_HALF_WINDOW + 2:
        logger.error(f"Trajectory {TRAJECTORY_FILE} too short for the zoom window")
        sys.exit(1)
    logger.info(f"Loaded {len(coords)} fixes from {TRAJECTORY_FILE}")

    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in coords],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    xs = points_gdf.geometry.x.values
    ys = points_gdf.geometry.y.values

    window_size = 2 * ZOOM_HALF_WINDOW
    if ZOOM_AUTO_SELECT:
        # Largest-displacement window so the zoom lands on a moving sub-segment.
        step = np.hypot(np.diff(xs), np.diff(ys))
        window_sums = np.convolve(step, np.ones(window_size - 1), mode="valid")
        best_start = int(np.argmax(window_sums))
        centre = best_start + ZOOM_HALF_WINDOW
        logger.info(f"Auto-selected zoom centre at index {centre} "
                    f"(window displacement {window_sums[best_start]:.0f} m)")
    else:
        centre = int(len(coords) * ZOOM_CENTER_FRACTION)
    zoom_start = max(0, centre - ZOOM_HALF_WINDOW)
    zoom_end = min(len(coords), centre + ZOOM_HALF_WINDOW)
    zx = xs[zoom_start:zoom_end]
    zy = ys[zoom_start:zoom_end]

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    ax_main.plot(xs, ys, color=COLOR_LINE, linewidth=1.4, alpha=0.9, zorder=2)
    ax_main.scatter(
        [xs[0]], [ys[0]],
        s=140, marker="^", color=COLOR_START,
        edgecolor="white", linewidth=1.4, zorder=4,
    )
    ax_main.scatter(
        [xs[-1]], [ys[-1]],
        s=130, marker="s", color=COLOR_END,
        edgecolor="white", linewidth=1.4, zorder=4,
    )

    # Force square aspect so the inset frame is not distorted relative to (b).
    span = max((zx.max() - zx.min()), (zy.max() - zy.min()))
    pad = span * 0.10 + 25
    cx = (zx.min() + zx.max()) / 2
    cy = (zy.min() + zy.max()) / 2
    half = span / 2 + pad
    rect_x0 = cx - half
    rect_y0 = cy - half
    rect_w = 2 * half
    rect_h = 2 * half
    ax_main.add_patch(Rectangle(
        (rect_x0, rect_y0), rect_w, rect_h,
        fill=False, edgecolor=COLOR_FRAME, linewidth=1.3,
        linestyle=(0, (4, 3)), zorder=5,
    ))

    try:
        ctx.add_basemap(ax_main, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Main basemap failed: {e}")

    ax_main.set_axis_off()
    ax_main.set_title("(a) Full trajectory", loc="left", fontsize=12,
                      fontweight="medium", pad=8)

    ax_zoom.plot(zx, zy, color=COLOR_LINE, linewidth=2.4, alpha=1.0, zorder=2,
                 solid_capstyle="round")
    ax_zoom.scatter(
        zx, zy,
        s=38, color=COLOR_DOT, alpha=1.0, zorder=3,
        edgecolor="white", linewidth=0.6,
    )
    ax_zoom.set_xlim(rect_x0, rect_x0 + rect_w)
    ax_zoom.set_ylim(rect_y0, rect_y0 + rect_h)

    try:
        ctx.add_basemap(ax_zoom, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Zoom basemap failed: {e}")

    # Overlay rectangle: spine styling is unreliable on top of contextily basemaps.
    for spine in ax_zoom.spines.values():
        spine.set_visible(False)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.add_patch(Rectangle(
        (rect_x0, rect_y0), rect_w, rect_h,
        fill=False, edgecolor=COLOR_FRAME, linewidth=1.4,
        linestyle="--", zorder=10, clip_on=False,
    ))
    ax_zoom.set_title("(b) Zoomed sub-segment", loc="left", fontsize=12,
                      fontweight="medium", pad=8)

    legend_elems = [
        Line2D([0], [0], color=COLOR_LINE, lw=2.0,
               label="Reconstructed path"),
        Line2D([0], [0], marker="o", lw=0, color=COLOR_DOT,
               markeredgecolor="white", markeredgewidth=0.8,
               markersize=8, label="GPS fix"),
        Line2D([0], [0], marker="^", lw=0, color=COLOR_START,
               markeredgecolor="white", markeredgewidth=1.0,
               markersize=10, label="Start"),
        Line2D([0], [0], marker="s", lw=0, color=COLOR_END,
               markeredgecolor="white", markeredgewidth=1.0,
               markersize=10, label="End"),
    ]
    ax_main.legend(
        handles=legend_elems, loc="lower left",
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fontsize=9.5,
    )

    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
