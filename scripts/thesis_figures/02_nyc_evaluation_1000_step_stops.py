"""Heatmap-style map of all STEP-detected stops on NYC_Evaluation_1000.

Runs the online STEP segmenter on every evaluation trajectory at the
calibrated defaults and plots each stop centroid as a low-alpha dot on a
CartoDB Positron basemap. Stops are cached; set FORCE_RECOMPUTE=True to refresh.
"""

# ruff: noqa: E402

import csv
import logging
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from core.segment import Stop
from core.stream import TrajectoryStream
from engines.step import STEPSegmenter

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(project_root, "data", "raw", "NYC_Evaluation_1000")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")
STOP_CACHE = os.path.join(OUTPUT_ROOT, f"{SCRIPT_NAME}_cache.csv")
FORCE_RECOMPUTE = False

DOT_COLOR = "#7f0000"
DOT_ALPHA = 0.10
DOT_SIZE = 6
FIG_SIZE = (14, 14)
DPI = 400


def compute_stops(logger: logging.Logger) -> list[dict]:
    """Run STEP on every CSV in DATA_DIR and return one record per stop."""
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    logger.info(f"Found {len(csv_files)} trajectory files in {DATA_DIR}")

    records: list[dict] = []
    for idx, fname in enumerate(csv_files, 1):
        obj_id = fname.replace(".csv", "")
        path = os.path.join(DATA_DIR, fname)

        stream = TrajectoryStream(
            filepath=path,
            col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
            default_obj_id=obj_id,
        )
        points = list(stream.stream())
        if len(points) < 2:
            logger.warning(f"[{idx}/{len(csv_files)}] {obj_id}: skipped (<2 points)")
            continue

        segmenter = STEPSegmenter(
            max_eps=STOP_MAX_EPS_METERS,
            min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        )
        segments = segmenter.process(points)

        n_stops = 0
        for seg in segments:
            if isinstance(seg, Stop) and seg.centroid is not None:
                records.append({
                    "obj_id": obj_id,
                    "lat": seg.centroid.lat,
                    "lon": seg.centroid.lon,
                    "start_time": seg.start_time.isoformat(),
                    "end_time": seg.end_time.isoformat(),
                })
                n_stops += 1

        logger.info(f"[{idx}/{len(csv_files)}] {obj_id}: {len(points)} pts -> {n_stops} stops")

    logger.info(f"Total stops detected: {len(records)}")
    return records


def write_cache(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["obj_id", "lat", "lon", "start_time", "end_time"]
        )
        writer.writeheader()
        writer.writerows(records)


def read_cache(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def plot_stops(records: list[dict], logger: logging.Logger) -> None:
    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import Point as ShapelyPoint
    except ImportError as e:
        logger.error(f"Required GIS dependency missing: {e}")
        sys.exit(1)

    geoms = [ShapelyPoint(float(r["lon"]), float(r["lat"])) for r in records]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    gdf.plot(
        ax=ax,
        color=DOT_COLOR,
        markersize=DOT_SIZE,
        alpha=DOT_ALPHA,
        linewidth=0,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("nyc_evaluation_step_stops")

    if not FORCE_RECOMPUTE and os.path.exists(STOP_CACHE):
        logger.info(f"Loading cached stops from {STOP_CACHE}")
        records = read_cache(STOP_CACHE)
        logger.info(f"Loaded {len(records)} cached stops")
    else:
        records = compute_stops(logger)
        write_cache(records, STOP_CACHE)
        logger.info(f"Cached {len(records)} stops to {STOP_CACHE}")

    if not records:
        logger.error("No stops to plot.")
        return

    plot_stops(records, logger)


if __name__ == "__main__":
    main()
