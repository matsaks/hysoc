"""Side-by-side overview maps of all NYC and SF Centre trajectories.

Every trajectory in the full 1,100-trajectory set per region is rendered as a
low-alpha polyline on a CartoDB Positron basemap so overlapping segments
accumulate into a density heatmap of road coverage.
"""

# ruff: noqa: E402

import logging
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from _latex_style import apply_latex_style

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RAW_ROOT = os.path.join(project_root, "data", "raw")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

# Each panel uses the same projected extent (Web Mercator metres) so the two
# map boxes look identical; `center` is (lat, lon) of the panel midpoint.
PANEL_EXTENT_M: tuple[float, float] = (12000.0, 10000.0)

DATASETS: list[dict] = [
    {
        "label": "NYC",
        "subdirs": ["NYC_Calibration_100", "NYC_Evaluation_1000"],
        "center": (40.7405, -73.970),
    },
    {
        "label": "SF Centre",
        "subdirs": ["SanFranCentre_Calibration_100", "SanFranCentre_Evaluation_1000"],
        "center": (37.7825, -122.4325),
    },
]

LINE_COLOR = "#08306b"
# Raise alpha to make sparse trajectories visible; routes saturate near ~0.25.
LINE_ALPHA = 0.12
LINE_WIDTH = 0.7
PANEL_SIZE = (8, 8)
DPI = 600
BASEMAP_ZOOM = 15       # +1 doubles tile detail and ~4x network/memory


def load_trajectory(filepath: str) -> list[tuple[float, float]]:
    import csv

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


def collect_geometries(subdirs: list[str], logger: logging.Logger):
    from shapely.geometry import LineString

    records = []
    for subdir in subdirs:
        data_dir = os.path.join(RAW_ROOT, subdir)
        if not os.path.isdir(data_dir):
            logger.warning(f"Missing dataset directory: {data_dir}")
            continue
        csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
        logger.info(f"{subdir}: {len(csv_files)} files")
        for fname in csv_files:
            coords = load_trajectory(os.path.join(data_dir, fname))
            if len(coords) < 2:
                continue
            records.append(
                {"obj_id": fname.replace(".csv", ""), "geometry": LineString(coords)}
            )
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("dataset_overview_maps")

    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.error(f"Required GIS dependency missing: {e}")
        sys.exit(1)

    apply_latex_style(use_latex=True)

    n_panels = len(DATASETS)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(PANEL_SIZE[0] * n_panels, PANEL_SIZE[1])
    )
    if n_panels == 1:
        axes = [axes]

    from shapely.geometry import Point as ShapelyPoint

    half_w, half_h = PANEL_EXTENT_M[0] / 2.0, PANEL_EXTENT_M[1] / 2.0

    for ax, spec in zip(axes, DATASETS):
        label = spec["label"]
        records = collect_geometries(spec["subdirs"], logger)
        if not records:
            logger.warning(f"{label}: no valid trajectories")
            ax.set_title(label)
            ax.set_axis_off()
            continue

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=3857)
        gdf.plot(ax=ax, color=LINE_COLOR, linewidth=LINE_WIDTH, alpha=LINE_ALPHA)

        center_lat, center_lon = spec["center"]
        cx, cy = (
            gpd.GeoSeries([ShapelyPoint(center_lon, center_lat)], crs="EPSG:4326")
            .to_crs(epsg=3857)
            .geometry.iloc[0]
            .coords[0]
        )
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)
        ax.set_aspect("equal")

        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=BASEMAP_ZOOM)
        except Exception as e:
            logger.warning(f"{label}: basemap failed: {e}")

        ax.set_title(label, fontsize=14)
        ax.set_axis_off()
        logger.info(f"{label}: plotted {len(records)} trajectories")

    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
