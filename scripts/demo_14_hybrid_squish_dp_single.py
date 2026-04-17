# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from core.point import Point
from core.segment import Move, Stop
from eval import calculate_sed_stats
from engines.dp import DouglasPeuckerCompressor
from engines.squish_dp import (
    HybridSquishDPCompressor,
    HybridSquishDPConfig,
)
from engines.squish import SquishCompressor
from engines.stop_compressor import CompressedStop, StopCompressor
from oracle.oracleDP import OracleDP
from oracle.oracleG import OracleG


DEFAULT_SUBSET = "subset_50"
DEFAULT_OBJ_ID = "7679671"
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS


def load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    points: List[Point] = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory CSV not found: {filepath}")

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


def reconstruct_for_sed(items: List[object]) -> Tuple[List[Point], int]:
    """
    Reconstruct the time-ordered compressed point stream used by `calculate_sed_stats`.

    Compression ratio counts:
    - STOP as 1 stored point (centroid)
    - MOVE as len(move.points) stored points
    """
    sed_stream: List[Point] = []
    stored_points = 0

    for item in items:
        if isinstance(item, CompressedStop):
            p_start = Point(
                lat=item.centroid.lat,
                lon=item.centroid.lon,
                timestamp=item.start_time,
                obj_id=item.centroid.obj_id,
            )
            p_end = Point(
                lat=item.centroid.lat,
                lon=item.centroid.lon,
                timestamp=item.end_time,
                obj_id=item.centroid.obj_id,
            )
            sed_stream.extend([p_start, p_end])
            stored_points += 1
        elif isinstance(item, Move):
            sed_stream.extend(item.points)
            stored_points += len(item.points)
    return sed_stream, stored_points


def plot_segment_branch(
    title: str,
    raw_points: List[Point],
    squish_points: List[Point],
    hybrid_points: List[Point],
    dp_points: List[Point],
    out_path: str,
) -> None:
    """
    Map-based visualization using geopandas/shapely/contextily when available.

    Produces a 2x2 grid:
    - Raw trajectory
    - SQUISH compressed
    - Hybrid (SQUISH+DP)
    - DP oracle
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString, Point as ShapelyPoint
        import contextily as ctx
    except ImportError:
        # Fallback: separate matplotlib subplots (no basemap).
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        def plot_series(ax, points, color, marker, label):
            if not points:
                return
            ax.plot([p.lon for p in points], [p.lat for p in points], color=color, alpha=0.6)
            ax.scatter([p.lon for p in points], [p.lat for p in points], color=color, s=10, marker=marker, label=label)

        plot_series(axes[0], raw_points, "gray", ".", "Raw")
        plot_series(axes[1], squish_points, "red", "o", "SQUISH")
        plot_series(axes[2], hybrid_points, "green", "x", "Hybrid")
        plot_series(axes[3], dp_points, "blue", "^", "DP")

        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.set_axis_off()

        axes[0].set_title(f"{title}\nRaw")
        axes[1].set_title("SQUISH")
        axes[2].set_title("Hybrid (SQUISH+DP)")
        axes[3].set_title("DP Oracle")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    def to_line(points: List[Point]):
        if not points:
            return None
        if len(points) < 2:
            return ShapelyPoint(points[0].lon, points[0].lat)
        return LineString([(p.lon, p.lat) for p in points])

    def to_points(points: List[Point]):
        return [ShapelyPoint(p.lon, p.lat) for p in points] if points else []

    raw_line = to_line(raw_points)
    squish_line = to_line(squish_points)
    hybrid_line = to_line(hybrid_points)
    dp_line = to_line(dp_points)

    raw_pts = to_points(raw_points)
    squish_pts = to_points(squish_points)
    hybrid_pts = to_points(hybrid_points)
    dp_pts = to_points(dp_points)

    # Create GeoDataFrames in EPSG:4326 then reproject for basemap.
    raw_line_gdf = gpd.GeoDataFrame([{"geometry": raw_line, "type": "RawLine"}], crs="EPSG:4326")
    raw_pts_gdf = gpd.GeoDataFrame([{"geometry": p, "type": "RawPoint"} for p in raw_pts], crs="EPSG:4326")

    squish_line_gdf = gpd.GeoDataFrame([{"geometry": squish_line, "type": "SQUISHLine"}], crs="EPSG:4326")
    squish_pts_gdf = gpd.GeoDataFrame([{"geometry": p, "type": "SQUISHPoint"} for p in squish_pts], crs="EPSG:4326")

    hybrid_line_gdf = gpd.GeoDataFrame([{"geometry": hybrid_line, "type": "HybridLine"}], crs="EPSG:4326")
    hybrid_pts_gdf = gpd.GeoDataFrame([{"geometry": p, "type": "HybridPoint"} for p in hybrid_pts], crs="EPSG:4326")

    dp_line_gdf = gpd.GeoDataFrame([{"geometry": dp_line, "type": "DPLine"}], crs="EPSG:4326")
    dp_pts_gdf = gpd.GeoDataFrame([{"geometry": p, "type": "DPPoint"} for p in dp_pts], crs="EPSG:4326")

    # Use per-panel basemap extents (same approach as demo_13).
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax_raw, ax_squish, ax_hybrid, ax_dp = axes.ravel()

    def plot_panel(ax, raw_line_part, raw_pts_part, comp_line, comp_pts, comp_color, comp_label):
        raw_line_part.plot(ax=ax, color="gray", linewidth=2, alpha=0.5)
        if not raw_pts_part.empty:
            raw_pts_part.plot(ax=ax, color="gray", markersize=5, alpha=0.35)

        comp_line.plot(ax=ax, color=comp_color, linewidth=2.2, alpha=0.9)
        if not comp_pts.empty:
            comp_pts.plot(ax=ax, color=comp_color, markersize=12, alpha=0.9)

        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass

        ax.set_axis_off()
        ax.set_title(comp_label)

    raw_line_web = raw_line_gdf.to_crs(epsg=3857)
    raw_pts_web = raw_pts_gdf.to_crs(epsg=3857)
    squish_line_web = squish_line_gdf.to_crs(epsg=3857)
    squish_pts_web = squish_pts_gdf.to_crs(epsg=3857)
    hybrid_line_web = hybrid_line_gdf.to_crs(epsg=3857)
    hybrid_pts_web = hybrid_pts_gdf.to_crs(epsg=3857)
    dp_line_web = dp_line_gdf.to_crs(epsg=3857)
    dp_pts_web = dp_pts_gdf.to_crs(epsg=3857)

    # Panel titles: keep the main `title` as prefix on Raw.
    plot_panel(ax_raw, raw_line_web, raw_pts_web, raw_line_web, raw_pts_web, "black", f"{title}\nRaw")
    plot_panel(ax_squish, raw_line_web, raw_pts_web, squish_line_web, squish_pts_web, "red", "SQUISH")
    plot_panel(ax_hybrid, raw_line_web, raw_pts_web, hybrid_line_web, hybrid_pts_web, "green", "Hybrid (SQUISH+DP)")
    plot_panel(ax_dp, raw_line_web, raw_pts_web, dp_line_web, dp_pts_web, "blue", "DP Oracle")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_full_run_squish(
    out_path: str,
    raw_points: List[Point],
    squish_points: List[Point],
    epsilon_capacity: int,
) -> None:
    """
    Demo plot: SQUISH applied to the entire trajectory (no STOP/MOVE segmentation).

    Produces a 1x2 figure (Raw vs Full-Run SQUISH) with a basemap when geopandas/contextily are available.
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString, Point as ShapelyPoint
        import contextily as ctx
    except ImportError:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].plot([p.lon for p in raw_points], [p.lat for p in raw_points], color="gray", alpha=0.4)
        axes[0].scatter([p.lon for p in raw_points], [p.lat for p in raw_points], s=5, color="gray", alpha=0.4)
        axes[0].set_axis_off()
        axes[0].set_title("Raw")

        axes[1].plot([p.lon for p in squish_points], [p.lat for p in squish_points], color="red", alpha=0.9)
        axes[1].scatter([p.lon for p in squish_points], [p.lat for p in squish_points], s=20, color="red", alpha=0.9)
        axes[1].set_axis_off()
        axes[1].set_title(f"Full-run SQUISH (cap={epsilon_capacity})")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    def to_line(points: List[Point]):
        if len(points) < 2:
            return ShapelyPoint(points[0].lon, points[0].lat)
        return LineString([(p.lon, p.lat) for p in points])

    raw_line = to_line(raw_points)
    squish_line = to_line(squish_points)

    raw_pts = [ShapelyPoint(p.lon, p.lat) for p in raw_points]
    squish_pts = [ShapelyPoint(p.lon, p.lat) for p in squish_points]

    raw_line_gdf = gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326")
    raw_pts_gdf = gpd.GeoDataFrame([{"geometry": p} for p in raw_pts], crs="EPSG:4326")

    squish_line_gdf = gpd.GeoDataFrame([{"geometry": squish_line}], crs="EPSG:4326")
    squish_pts_gdf = gpd.GeoDataFrame([{"geometry": p} for p in squish_pts], crs="EPSG:4326")

    raw_line_web = raw_line_gdf.to_crs(epsg=3857)
    raw_pts_web = raw_pts_gdf.to_crs(epsg=3857)
    squish_line_web = squish_line_gdf.to_crs(epsg=3857)
    squish_pts_web = squish_pts_gdf.to_crs(epsg=3857)

    fig, (ax_raw, ax_squish) = plt.subplots(1, 2, figsize=(16, 7))

    raw_line_web.plot(ax=ax_raw, color="gray", linewidth=2, alpha=0.6)
    raw_pts_web.plot(ax=ax_raw, color="gray", markersize=6, alpha=0.5)
    ax_raw.set_axis_off()
    ax_raw.set_title("Raw trajectory")

    squish_line_web.plot(ax=ax_squish, color="red", linewidth=2.2, alpha=0.95)
    squish_pts_web.plot(ax=ax_squish, color="red", markersize=16, alpha=0.9)
    ax_squish.set_axis_off()
    ax_squish.set_title(f"Full-run SQUISH (cap={epsilon_capacity})")

    for ax in (ax_raw, ax_squish):
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: Hybrid SQUISH + DP move compression (HYSOC-G hybrid)."
    )
    # Note: the input CSV selection (subset + obj_id) is intentionally kept as constants
    # so you don't have to pass file paths/ids as command-line args.
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--dp-refine-when-evictions", action="store_true")
    parser.add_argument("--output-root", default=os.path.join("data", "processed", "demo_14_hybrid_squish_dp"))
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, "data", "raw", DEFAULT_SUBSET)
    data_path = os.path.join(subset_dir, f"{DEFAULT_OBJ_ID}.csv")

    print(f"Loading trajectory: {data_path}")
    trajectory = load_trajectory(data_path, obj_id=str(DEFAULT_OBJ_ID))
    if len(trajectory) < 2:
        raise ValueError(f"Not enough points in trajectory ({len(trajectory)}).")
    print(f"Loaded {len(trajectory)} points.")

    print("Segmenting (OracleG)...")
    oracle = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    segments = oracle.process(trajectory)
    print(f"Segmentation produced {len(segments)} segments.")

    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)

    # --- Demo experiment: apply SQUISH directly to the full trajectory ---
    # This is intentionally "without stop segmentation": we ignore Module I outputs
    # and treat the whole stream as one MOVE-like polyline.
    squish_full_points = squish.compress(trajectory, capacity=args.buffer_capacity)

    hybrid = HybridSquishDPCompressor(
        HybridSquishDPConfig(
            capacity=args.buffer_capacity,
            dp_epsilon_meters=args.dp_epsilon_meters,
            dp_refine_when_evictions=args.dp_refine_when_evictions,
        )
    )

    dp_oracle = OracleDP(epsilon_meters=args.dp_epsilon_meters)

    processed_squish: List[object] = []
    processed_hybrid: List[object] = []
    processed_dp: List[object] = []

    short_moves = 0
    long_moves = 0
    first_short_move: Optional[List[Point]] = None
    first_long_move: Optional[List[Point]] = None

    for seg in segments:
        if isinstance(seg, Stop):
            stop_item = stop_compressor.compress(seg.points)
            processed_squish.append(stop_item)
            processed_hybrid.append(stop_item)
            processed_dp.append(stop_item)
            continue

        if isinstance(seg, Move):
            move_points = seg.points
            if len(move_points) <= args.buffer_capacity:
                short_moves += 1
                if first_short_move is None:
                    first_short_move = move_points
            else:
                long_moves += 1
                if first_long_move is None:
                    first_long_move = move_points

            squish_points = squish.compress(move_points, capacity=args.buffer_capacity)
            hybrid_points = hybrid.compress(
                move_points, capacity=args.buffer_capacity, dp_epsilon_meters=args.dp_epsilon_meters
            )
            dp_points = dp_oracle.process(seg)

            processed_squish.append(Move(points=squish_points))
            processed_hybrid.append(Move(points=hybrid_points))
            processed_dp.append(Move(points=dp_points))

    print(f"Move segments: short={short_moves}, long={long_moves} (buffer={args.buffer_capacity})")

    sed_squish, stored_squish = reconstruct_for_sed(processed_squish)
    sed_hybrid, stored_hybrid = reconstruct_for_sed(processed_hybrid)
    sed_dp, stored_dp = reconstruct_for_sed(processed_dp)

    cr_squish = len(trajectory) / max(1, stored_squish)
    cr_hybrid = len(trajectory) / max(1, stored_hybrid)
    cr_dp = len(trajectory) / max(1, stored_dp)

    sed_stats_squish = calculate_sed_stats(trajectory, sed_squish)
    sed_stats_hybrid = calculate_sed_stats(trajectory, sed_hybrid)
    sed_stats_dp = calculate_sed_stats(trajectory, sed_dp)

    full_run_squish_cr = len(trajectory) / max(1, len(squish_full_points))
    full_run_sed_stats_squish = calculate_sed_stats(trajectory, squish_full_points)

    print("\nMetrics (SED):")
    print(f" SQUISH: CR={cr_squish:.3f}, avg={sed_stats_squish['average_sed']:.3f}m, max={sed_stats_squish['max_sed']:.3f}m")
    print(f" HYBRID: CR={cr_hybrid:.3f}, avg={sed_stats_hybrid['average_sed']:.3f}m, max={sed_stats_hybrid['max_sed']:.3f}m")
    print(f" DP ORACLE: CR={cr_dp:.3f}, avg={sed_stats_dp['average_sed']:.3f}m, max={sed_stats_dp['max_sed']:.3f}m")
    print(
        f" FULL-RUN SQUISH: CR={full_run_squish_cr:.3f}, avg={full_run_sed_stats_squish['average_sed']:.3f}m, max={full_run_sed_stats_squish['max_sed']:.3f}m"
    )

    # Outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, f"{timestamp}_{DEFAULT_OBJ_ID}")
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "subset": DEFAULT_SUBSET,
        "obj_id": DEFAULT_OBJ_ID,
        "buffer_capacity": args.buffer_capacity,
        "dp_epsilon_meters": args.dp_epsilon_meters,
        "dp_refine_when_evictions": args.dp_refine_when_evictions,
        "move_segments": {"short": short_moves, "long": long_moves},
        "squish": {
            "compression_ratio": cr_squish,
            "stored_points": stored_squish,
            "average_sed_m": sed_stats_squish["average_sed"],
            "max_sed_m": sed_stats_squish["max_sed"],
            "rmse_m": sed_stats_squish["rmse"],
        },
        "hybrid": {
            "compression_ratio": cr_hybrid,
            "stored_points": stored_hybrid,
            "average_sed_m": sed_stats_hybrid["average_sed"],
            "max_sed_m": sed_stats_hybrid["max_sed"],
            "rmse_m": sed_stats_hybrid["rmse"],
        },
        "full_run_squish": {
            "compression_ratio": full_run_squish_cr,
            "stored_points": len(squish_full_points),
            "average_sed_m": full_run_sed_stats_squish["average_sed"],
            "max_sed_m": full_run_sed_stats_squish["max_sed"],
            "rmse_m": full_run_sed_stats_squish["rmse"],
        },
        "dp_oracle": {
            "compression_ratio": cr_dp,
            "stored_points": stored_dp,
            "average_sed_m": sed_stats_dp["average_sed"],
            "max_sed_m": sed_stats_dp["max_sed"],
            "rmse_m": sed_stats_dp["rmse"],
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", newline="") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    full_run_plot_path = os.path.join(out_dir, "full_run_squish.png")
    plot_full_run_squish(
        out_path=full_run_plot_path,
        raw_points=trajectory,
        squish_points=squish_full_points,
        epsilon_capacity=args.buffer_capacity,
    )
    print(f"Saved full-run SQUISH plot: {full_run_plot_path}")

    # Segment branch showcase (matplotlib-only).
    if first_short_move is not None:
        squish_points = squish.compress(first_short_move, capacity=args.buffer_capacity)
        hybrid_points = hybrid.compress(
            first_short_move, capacity=args.buffer_capacity, dp_epsilon_meters=args.dp_epsilon_meters
        )
        dp_points = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters).compress(first_short_move)
        out_path = os.path.join(out_dir, "branch_short_move.png")
        plot_segment_branch(
            title=f"Short move branch (len={len(first_short_move)} <= cap={args.buffer_capacity})",
            raw_points=first_short_move,
            squish_points=squish_points,
            hybrid_points=hybrid_points,
            dp_points=dp_points,
            out_path=out_path,
        )

    if first_long_move is not None:
        squish_points = squish.compress(first_long_move, capacity=args.buffer_capacity)
        hybrid_points = hybrid.compress(
            first_long_move, capacity=args.buffer_capacity, dp_epsilon_meters=args.dp_epsilon_meters
        )
        dp_points = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters).compress(first_long_move)
        out_path = os.path.join(out_dir, "branch_long_move.png")
        plot_segment_branch(
            title=f"Long move branch (len={len(first_long_move)} > cap={args.buffer_capacity})",
            raw_points=first_long_move,
            squish_points=squish_points,
            hybrid_points=hybrid_points,
            dp_points=dp_points,
            out_path=out_path,
        )

    print(f"Segment branch plots saved under: {out_dir}")


if __name__ == "__main__":
    main()

