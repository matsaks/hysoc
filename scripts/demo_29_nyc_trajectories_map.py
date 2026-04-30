"""Plot all NYC_Trajectories_Polygon trajectories on a single CartoDB.Positron map."""

import os
import sys
import csv
import logging
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

DATA_DIR = os.path.join(project_root, "data", "raw", "NYC_Top_1000_Longest")
OUTPUT_DIR = os.path.join(project_root, "data", "processed", "demo_29_nyc_trajectories_map")


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
    logger = logging.getLogger("nyc_trajectories_map")

    csv_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith(".csv")
    )
    logger.info(f"Found {len(csv_files)} trajectory files in {DATA_DIR}")

    try:
        import geopandas as gpd
        from shapely.geometry import LineString, MultiPoint
        import contextily as ctx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError as e:
        logger.error(f"Required GIS dependency missing: {e}")
        sys.exit(1)

    records: list[dict] = []
    for fname in csv_files:
        obj_id = fname.replace(".csv", "")
        coords = load_trajectory(os.path.join(DATA_DIR, fname))
        if len(coords) < 2:
            logger.warning(f"Skipping {fname}: fewer than 2 points")
            continue
        records.append({"obj_id": obj_id, "geometry": LineString(coords)})

    logger.info(f"Loaded {len(records)} valid trajectories")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=3857)

    colors = cm.tab20(np.linspace(0, 1, len(gdf)))

    fig, ax = plt.subplots(figsize=(14, 14))

    for (_, row), color in zip(gdf.iterrows(), colors):
        gpd.GeoDataFrame([row], crs=gdf.crs).plot(
            ax=ax, color=color, linewidth=1.0, alpha=0.75
        )

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Basemap failed: {e}")

    ax.set_axis_off()
    ax.set_title(f"NYC Trajectories — {len(gdf)} vehicles", fontsize=14, pad=12)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{timestamp}_nyc_trajectories.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved map: {out_path}")


if __name__ == "__main__":
    main()
