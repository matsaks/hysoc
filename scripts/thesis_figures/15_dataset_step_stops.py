"""Side-by-side STEP-stop heatmaps for the NYC and SF Centre datasets.

Runs the online STEP segmenter at calibrated defaults on the full
1,100-trajectory sets per region and plots each stop centroid as a low-alpha
dot on the same projected extent as figure 14. Stops are cached; set
FORCE_RECOMPUTE=True to refresh.
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

from _latex_style import apply_latex_style
from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from core.segment import Stop
from core.stream import TrajectoryStream
from engines.step import STEPSegmenter

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RAW_ROOT = os.path.join(project_root, "data", "raw")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")
CACHE_PATH = os.path.join(OUTPUT_ROOT, f"{SCRIPT_NAME}_cache.csv")
FORCE_RECOMPUTE = False

PANEL_EXTENT_M: tuple[float, float] = (12000.0, 10000.0)

DATASETS: list[dict] = [
    {
        "label": "NYC",
        "key": "NYC",
        "subdirs": ["NYC_Calibration_100", "NYC_Evaluation_1000"],
        "center": (40.7405, -73.970),
    },
    {
        "label": "SF Centre",
        "key": "SF_Centre",
        "subdirs": ["SanFranCentre_Calibration_100", "SanFranCentre_Evaluation_1000"],
        "center": (37.7825, -122.4325),
    },
]

DOT_COLOR = "#7f0000"
# Raise for sparse stop visibility, lower to preserve the cluster gradient.
DOT_ALPHA = 0.40
DOT_SIZE = 8
PANEL_SIZE = (8, 8)
DPI = 600
BASEMAP_ZOOM = 15   # +1 doubles tile detail and ~4x network/memory


def compute_stops(logger: logging.Logger) -> list[dict]:
    records: list[dict] = []
    for spec in DATASETS:
        key, label = spec["key"], spec["label"]
        for subdir in spec["subdirs"]:
            data_dir = os.path.join(RAW_ROOT, subdir)
            if not os.path.isdir(data_dir):
                logger.warning(f"Missing dataset directory: {data_dir}")
                continue
            csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
            logger.info(f"{label} / {subdir}: {len(csv_files)} files")
            for idx, fname in enumerate(csv_files, 1):
                obj_id = fname.replace(".csv", "")
                stream = TrajectoryStream(
                    filepath=os.path.join(data_dir, fname),
                    col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
                    default_obj_id=obj_id,
                )
                points = list(stream.stream())
                if len(points) < 2:
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
                            "dataset": key,
                            "obj_id": obj_id,
                            "lat": seg.centroid.lat,
                            "lon": seg.centroid.lon,
                        })
                        n_stops += 1
                if idx % 100 == 0 or idx == len(csv_files):
                    logger.info(
                        f"  [{idx}/{len(csv_files)}] running total stops: "
                        f"{sum(1 for r in records if r['dataset'] == key)}"
                    )
    return records


def write_cache(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "obj_id", "lat", "lon"])
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

    apply_latex_style(use_latex=True)

    by_key: dict[str, list[dict]] = {}
    for r in records:
        by_key.setdefault(r["dataset"], []).append(r)

    n_panels = len(DATASETS)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(PANEL_SIZE[0] * n_panels, PANEL_SIZE[1])
    )
    if n_panels == 1:
        axes = [axes]

    half_w, half_h = PANEL_EXTENT_M[0] / 2.0, PANEL_EXTENT_M[1] / 2.0

    for ax, spec in zip(axes, DATASETS):
        label, key = spec["label"], spec["key"]
        rows = by_key.get(key, [])

        center_lat, center_lon = spec["center"]
        cx, cy = (
            gpd.GeoSeries([ShapelyPoint(center_lon, center_lat)], crs="EPSG:4326")
            .to_crs(epsg=3857)
            .geometry.iloc[0]
            .coords[0]
        )

        if rows:
            geoms = [ShapelyPoint(float(r["lon"]), float(r["lat"])) for r in rows]
            gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326").to_crs(epsg=3857)
            gdf.plot(
                ax=ax,
                color=DOT_COLOR,
                markersize=DOT_SIZE,
                alpha=DOT_ALPHA,
                linewidth=0,
            )
            logger.info(f"{label}: plotted {len(rows)} stops")
        else:
            logger.warning(f"{label}: no stops found")

        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)
        ax.set_aspect("equal")

        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=BASEMAP_ZOOM)
        except Exception as e:
            logger.warning(f"{label}: basemap failed: {e}")

        ax.set_title(label, fontsize=14)
        ax.set_axis_off()

    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("dataset_step_stops")

    if not FORCE_RECOMPUTE and os.path.exists(CACHE_PATH):
        logger.info(f"Loading cached stops from {CACHE_PATH}")
        records = read_cache(CACHE_PATH)
        logger.info(f"Loaded {len(records)} cached stops")
    else:
        records = compute_stops(logger)
        write_cache(records, CACHE_PATH)
        logger.info(f"Cached {len(records)} stops to {CACHE_PATH}")

    if not records:
        logger.error("No stops to plot.")
        return

    plot_stops(records, logger)


if __name__ == "__main__":
    main()
