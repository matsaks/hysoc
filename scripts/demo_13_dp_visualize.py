import os
import sys
import csv
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.core.point import Point
from hysoc.metrics import calculate_sed_stats, calculate_compression_ratio
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor


# Defaults (used by this demo; change here for a different trajectory/epsilon).
DEFAULT_SUBSET = "London_Final_100"
DEFAULT_OBJ_ID = "5564434"
DEFAULT_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS


def load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    points: List[Point] = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dt = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue
            points.append(Point(lat=lat, lon=lon, timestamp=dt, obj_id=obj_id))
    return points


def compute_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 95))


def try_make_plot(
    out_path: str,
    original: List[Point],
    compressed: List[Point],
    epsilon_meters: float,
) -> None:
    """
    Creates a raw-vs-compressed overlay plot.

    Uses geopandas/shapely/contextily when available; otherwise does nothing.
    """

    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def to_lines(points: List[Point]):
        if len(points) < 2:
            return ShapelyPoint(points[0].lon, points[0].lat)
        return LineString([(p.lon, p.lat) for p in points])

    def to_points(points: List[Point]):
        # Plot as individual point markers (not a LineString) for clarity.
        return [ShapelyPoint(p.lon, p.lat) for p in points]

    raw_points_geom = to_points(original)
    dp_points_geom = to_points(compressed)

    def to_line(points: List[Point]):
        if len(points) < 2:
            return ShapelyPoint(points[0].lon, points[0].lat)
        return LineString([(p.lon, p.lat) for p in points])

    raw_line_geom = to_line(original)
    dp_line_geom = to_line(compressed)

    raw_gdf = gpd.GeoDataFrame(
        [{"type": "Raw", "geometry": g} for g in raw_points_geom],
        crs="EPSG:4326",
    )
    dp_gdf = gpd.GeoDataFrame(
        [{"type": "DP", "geometry": g} for g in dp_points_geom],
        crs="EPSG:4326",
    )

    raw_line_gdf = gpd.GeoDataFrame([{"type": "RawLine", "geometry": raw_line_geom}], crs="EPSG:4326")
    dp_line_gdf = gpd.GeoDataFrame([{"type": "DPLine", "geometry": dp_line_geom}], crs="EPSG:4326")

    raw_gdf_web = raw_gdf.to_crs(epsg=3857)
    dp_gdf_web = dp_gdf.to_crs(epsg=3857)
    raw_line_gdf_web = raw_line_gdf.to_crs(epsg=3857)
    dp_line_gdf_web = dp_line_gdf.to_crs(epsg=3857)

    fig, (ax_raw, ax_dp) = plt.subplots(1, 2, figsize=(16, 9))

    # Draw connecting line first (neutral color), then draw colored points on top.
    raw_line_gdf_web.plot(ax=ax_raw, color="gray", linewidth=2, alpha=0.65)
    raw_gdf_web.plot(ax=ax_raw, color="blue", markersize=6, alpha=0.6)
    ax_raw.set_axis_off()
    ax_raw.set_title("Raw trajectory")

    dp_line_gdf_web.plot(ax=ax_dp, color="gray", linewidth=2, alpha=0.65)
    dp_gdf_web.plot(ax=ax_dp, color="red", markersize=18, alpha=0.85)
    ax_dp.set_axis_off()
    ax_dp.set_title(f"DP compressed (epsilon={epsilon_meters} m)")

    # Basemap per axis so each panel uses its own extent
    for ax in (ax_raw, ax_dp):
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize offline geometric benchmark: Douglas-Peucker (DP) compression on one trajectory."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "processed", "demo_13_dp_visualize"),
        help="Output root directory (default: data/processed/demo_13_dp_visualize).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING (default: INFO)")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid --log-level: {args.log_level}")

    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("dp_visualize")

    subset_dir = os.path.join(project_root, "data", "raw", DEFAULT_SUBSET)
    filepath = os.path.join(subset_dir, f"{DEFAULT_OBJ_ID}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory CSV not found: {filepath}")

    logger.info(f"Loading trajectory: {filepath}")
    original_points = load_trajectory(filepath, obj_id=str(DEFAULT_OBJ_ID))
    if len(original_points) < 2:
        raise ValueError(f"Not enough points in trajectory ({len(original_points)}).")
    logger.info(f"Loaded {len(original_points)} points.")

    epsilon_meters = float(DEFAULT_EPSILON_METERS)
    compressor = DouglasPeuckerCompressor(epsilon_meters=epsilon_meters)
    compressed_points = compressor.compress(original_points)
    logger.info(f"DP compressed to {len(compressed_points)} points (epsilon={epsilon_meters} m).")

    cr = calculate_compression_ratio(original_points, compressed_points)
    sed_stats = calculate_sed_stats(original_points, compressed_points)
    sed_errors = sed_stats.get("sed_errors", [])
    p95_sed = compute_p95(sed_errors)

    # Outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(project_root, args.output_dir, f"{timestamp}_{DEFAULT_OBJ_ID}")
    os.makedirs(out_root, exist_ok=True)

    raw_csv_path = os.path.join(out_root, "raw_trajectory.csv")
    comp_csv_path = os.path.join(out_root, "compressed_trajectory.csv")
    metrics_path = os.path.join(out_root, "metrics.json")
    plot_path = os.path.join(out_root, "plot.png")

    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time"])
        for p in original_points:
            writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])

    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time"])
        for p in compressed_points:
            writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])

    metrics: Dict[str, object] = {
        "subset": DEFAULT_SUBSET,
        "obj_id": str(DEFAULT_OBJ_ID),
        "epsilon_meters": epsilon_meters,
        "cr": float(cr),
        "original_points": int(len(original_points)),
        "compressed_points": int(len(compressed_points)),
        "average_sed_m": float(sed_stats.get("average_sed", 0.0)),
        "max_sed_m": float(sed_stats.get("max_sed", 0.0)),
        "rmse_m": float(sed_stats.get("rmse", 0.0)),
        "p95_sed_m": float(p95_sed),
    }

    with open(metrics_path, "w", newline="") as f:
        json.dump(metrics, f, indent=2)

    # Optional plot (GIS libs may be missing)
    try:
        try_make_plot(plot_path, original_points, compressed_points, epsilon_meters=epsilon_meters)
    except Exception as e:
        logger.warning(f"Plot generation failed (continuing without plot.png): {e}")

    logger.info(f"Saved raw trajectory: {raw_csv_path}")
    logger.info(f"Saved DP compressed trajectory: {comp_csv_path}")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Plot path (may be missing if GIS deps not installed): {plot_path}")


if __name__ == "__main__":
    main()

