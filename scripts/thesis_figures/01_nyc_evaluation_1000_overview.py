"""Overview map of all NYC_Evaluation_1000 trajectories on a single basemap.

Renders every trajectory as a low-alpha polyline so overlapping segments
accumulate visually into a heatmap-like density of NYC coverage.
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

LINE_COLOR = "#08306b"
LINE_ALPHA = 0.05
LINE_WIDTH = 0.5
FIG_SIZE = (14, 14)
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
    logger = logging.getLogger("nyc_evaluation_overview")

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    logger.info(f"Found {len(csv_files)} trajectory files in {DATA_DIR}")

    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString
    except ImportError as e:
        logger.error(f"Required GIS dependency missing: {e}")
        sys.exit(1)

    records: list[dict] = []
    for fname in csv_files:
        coords = load_trajectory(os.path.join(DATA_DIR, fname))
        if len(coords) < 2:
            logger.warning(f"Skipping {fname}: fewer than 2 points")
            continue
        records.append({"obj_id": fname.replace(".csv", ""), "geometry": LineString(coords)})

    logger.info(f"Loaded {len(records)} valid trajectories")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    gdf.plot(
        ax=ax,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        alpha=LINE_ALPHA,
    )

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Basemap failed: {e}")

    ax.set_axis_off()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    logger.info(f"Saved map: {out_path}")


if __name__ == "__main__":
    main()
