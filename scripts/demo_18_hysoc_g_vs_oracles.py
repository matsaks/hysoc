# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from hysoc.constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from hysoc.core.point import Point
from hysoc.core.segment import Move, Stop
from hysoc.metrics import calculate_sed_stats
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor
from hysoc.modules.move_compression.squish import SquishCompressor
from hysoc.modules.segmentation.step import STEPSegmenter
from hysoc.modules.stop_compression.compressor import CompressedStop, StopCompressor
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_18_hysoc_g_vs_oracles")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS
DEFAULT_SUBSET_DIR = os.path.join("data", "raw", "subset_50")

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

def reconstruct_for_sed(items: List[object]) -> Tuple[List[Point], int]:
    sed_stream: List[Point] = []
    stored_points = 0
    for item in items:
        if isinstance(item, CompressedStop):
            p_start = Point(lat=item.centroid.lat, lon=item.centroid.lon, timestamp=item.start_time, obj_id=item.centroid.obj_id)
            p_end = Point(lat=item.centroid.lat, lon=item.centroid.lon, timestamp=item.end_time, obj_id=item.centroid.obj_id)
            sed_stream.extend([p_start, p_end])
            stored_points += 1
        elif isinstance(item, Move):
            sed_stream.extend(item.points)
            stored_points += len(item.points)
    return sed_stream, stored_points

def compute_segmented_metrics(original: List[Point], items: List[object], latency_us: float) -> Dict[str, Any]:
    sed_stream, stored_points = reconstruct_for_sed(items)
    cr = len(original) / max(1, stored_points)
    stats = calculate_sed_stats(original, sed_stream)
    sed_errors = stats.get("sed_errors", [])
    
    if not sed_errors:
        avg_sed, p95_sed, max_sed = 0.0, 0.0, 0.0
    else:
        arr = np.asarray(sed_errors, dtype=float)
        avg_sed = float(stats["average_sed"])
        p95_sed = float(np.percentile(arr, 95))
        max_sed = float(stats["max_sed"])
        
    return {
        "cr": cr,
        "stored_points": stored_points,
        "avg_sed_m": avg_sed,
        "p95_sed_m": p95_sed,
        "max_sed_m": max_sed,
        "latency_us_per_point": latency_us
    }

def plot_trajectory_comparison(
    out_path: str, 
    raw_points: List[Point], 
    plain_dp_items: List[object], 
    oracle_items: List[object], 
    hysoc_items: List[object], 
    title: str
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point as ShapelyPoint
    import contextily as ctx

    def items_to_geom(items: List[object]):
        move_pts = []
        stops = []
        for item in items:
            if isinstance(item, CompressedStop):
                stops.append(ShapelyPoint(item.centroid.lon, item.centroid.lat))
            elif isinstance(item, Move):
                for p in item.points:
                    move_pts.append(ShapelyPoint(p.lon, p.lat))
        return move_pts, stops

    raw_pts = [ShapelyPoint(p.lon, p.lat) for p in raw_points] if raw_points else []
    
    plain_dp_move_pts, plain_dp_stops = items_to_geom(plain_dp_items)
    oracle_move_pts, oracle_stops = items_to_geom(oracle_items)
    hysoc_move_pts, hysoc_stops = items_to_geom(hysoc_items)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.02)
    ax_raw, ax_plain, ax_oracle, ax_hysoc = axes.ravel()

    def plot_panel(ax, moves, stops, comp_color, is_raw=False):
        # 1. Background context (Gray)
        if raw_pts:
            gpd.GeoSeries(raw_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color="gray", markersize=5, alpha=0.35)

        # 2. Foreground Data (Prominent Marker Coloring)
        if is_raw and raw_pts:
            gpd.GeoSeries(raw_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color=comp_color, markersize=12, alpha=0.9)

        if moves and not is_raw:
            gpd.GeoSeries(moves, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color=comp_color, markersize=12, alpha=0.9, label="Move Point")
            
        if stops and not is_raw:
            gpd.GeoSeries(stops, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color="black", markersize=30, marker="o", zorder=5, label="Stop Centroid")
            
        if not is_raw and (moves or stops):
            ax.legend(loc="upper right")

        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass
        ax.set_axis_off()

    plot_panel(ax_raw, [], [], "black", is_raw=True)
    ax_raw.set_title(f"Raw Trajectory ({len(raw_points)} pts)", fontsize=14)

    plot_panel(ax_plain, plain_dp_move_pts, plain_dp_stops, "red")
    ax_plain.set_title("Plain DP (No Seg)", fontsize=14)

    plot_panel(ax_oracle, oracle_move_pts, oracle_stops, "blue")
    ax_oracle.set_title("Offline Oracle (STSS + DP)", fontsize=14)

    plot_panel(ax_hysoc, hysoc_move_pts, hysoc_stops, "green")
    ax_hysoc.set_title("HYSOC-G (STEP + SQUISH/DP)", fontsize=14)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 18: HYSOC-G vs Oracles Evaluation (Added Plain DP).")
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, DEFAULT_SUBSET_DIR)
    csv_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    csv_files.sort(key=lambda name: int(os.path.splitext(name)[0]) if os.path.splitext(name)[0].isdigit() else 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running HYSOC-G Evaluation on {len(csv_files)} trajectories in {subset_dir} ...")

    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    stss_oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )

    results = []

    for fname in csv_files:
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(subset_dir, fname)
        raw_points = load_trajectory(path, obj_id=obj_id)
        if len(raw_points) < 2:
            continue
            
        n_raw = len(raw_points)

        # ----------------------------------------------------
        # 1. Pipeline: Plain Fast DP (Treating whole trajectory as one move)
        # ----------------------------------------------------
        t0_plain = time.perf_counter()
        plain_dp_points = dp_compressor.compress(raw_points)
        processed_plain = [Move(points=plain_dp_points)]
        t1_plain = time.perf_counter()
        latency_us_plain = ((t1_plain - t0_plain) * 1e6) / n_raw
        metrics_plain = compute_segmented_metrics(raw_points, processed_plain, latency_us_plain)

        # ----------------------------------------------------
        # 2. Pipeline: Offline Oracle (STSS + DP)
        # ----------------------------------------------------
        t0_oracle = time.perf_counter()
        segments_stss = stss_oracle.process(raw_points)
        processed_oracle = []
        for seg in segments_stss:
            if isinstance(seg, Stop):
                processed_oracle.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                processed_oracle.append(Move(points=dp_compressor.compress(seg.points)))
        t1_oracle = time.perf_counter()
        latency_us_oracle = ((t1_oracle - t0_oracle) * 1e6) / n_raw
        metrics_oracle = compute_segmented_metrics(raw_points, processed_oracle, latency_us_oracle)

        # ----------------------------------------------------
        # 3. Pipeline: HYSOC-G (STEP + SQUISH/DP)
        # ----------------------------------------------------
        step_segmenter_local = STEPSegmenter(
            max_eps=STOP_MAX_EPS_METERS,
            min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        )
        t0_hysoc = time.perf_counter()
        segments_step = []
        for p in raw_points:
            segments_step.extend(step_segmenter_local.process_point(p))
        segments_step.extend(step_segmenter_local.flush())

        processed_hysoc = []
        for seg in segments_step:
            if isinstance(seg, Stop):
                processed_hysoc.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                squish_move = squish.compress(seg.points, capacity=args.buffer_capacity)
                dp_on_squish_move = dp_compressor.compress(squish_move)
                processed_hysoc.append(Move(points=dp_on_squish_move))
        
        t1_hysoc = time.perf_counter()
        latency_us_hysoc = ((t1_hysoc - t0_hysoc) * 1e6) / n_raw
        metrics_hysoc = compute_segmented_metrics(raw_points, processed_hysoc, latency_us_hysoc)

        rec = {
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            **{f"plain_{k}": v for k, v in metrics_plain.items()},
            **{f"oracle_{k}": v for k, v in metrics_oracle.items()},
            **{f"hysoc_{k}": v for k, v in metrics_hysoc.items()}
        }
        results.append(rec)
        
        # --- PER-FILE OUTPUTS ---
        obj_dir = os.path.join(out_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        
        with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
            json.dump(rec, f, indent=2)
            
        try:
            plot_trajectory_comparison(
                out_path=os.path.join(obj_dir, "trajectory_comparison.png"),
                raw_points=raw_points,
                plain_dp_items=processed_plain,
                oracle_items=processed_oracle,
                hysoc_items=processed_hysoc,
                title=f"Trajectory {obj_id} Comparison"
            )
        except Exception as e:
            print(f"Skipping plot for {obj_id}: {e}")

    # Aggregate CSV
    csv_path = os.path.join(out_dir, "demo18_evaluation_metrics.csv")
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved evaluation aggregate metrics: {csv_path}")

        # Also save aggregated summary mapping (mean & median)
        agg_metrics = {"mean": {}, "median": {}}
        keys_to_agg = [
            "plain_cr", "oracle_cr", "hysoc_cr", 
            "plain_avg_sed_m", "oracle_avg_sed_m", "hysoc_avg_sed_m", 
            "plain_latency_us_per_point", "oracle_latency_us_per_point", "hysoc_latency_us_per_point"
        ]
        for key in keys_to_agg:
            vals = [float(r[key]) for r in results]
            agg_metrics["mean"][key] = float(np.mean(vals))
            agg_metrics["median"][key] = float(np.median(vals))
        
        agg_path = os.path.join(out_dir, "agg_summary.json")
        with open(agg_path, "w", newline="") as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"Saved aggregated statistics: {agg_path}")

    # Summary Plotting
    if not results:
        return
    plain_cr = [r["plain_cr"] for r in results]
    oracle_cr = [r["oracle_cr"] for r in results]
    hysoc_cr = [r["hysoc_cr"] for r in results]
    
    plain_sed = [r["plain_avg_sed_m"] for r in results]
    oracle_sed = [r["oracle_avg_sed_m"] for r in results]
    hysoc_sed = [r["hysoc_avg_sed_m"] for r in results]
    
    plain_lat = [r["plain_latency_us_per_point"] for r in results]
    oracle_lat = [r["oracle_latency_us_per_point"] for r in results]
    hysoc_lat = [r["hysoc_latency_us_per_point"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['Plain DP', 'Offline Oracle', 'HYSOC-G']

    axes[0].boxplot([plain_cr, oracle_cr, hysoc_cr], tick_labels=labels, showmeans=True)
    axes[0].set_title("Compression Ratio (CR)")
    axes[0].set_ylabel("Compression Ratio (Raw/Compressed)")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([plain_sed, oracle_sed, hysoc_sed], tick_labels=labels, showmeans=True)
    axes[1].set_title("Information Loss (Mean SED)")
    axes[1].set_ylabel("Synchronized Euclidean Distance (meters)")
    axes[1].grid(True, alpha=0.3)

    axes[2].boxplot([plain_lat, oracle_lat, hysoc_lat], tick_labels=labels, showmeans=True)
    axes[2].set_title("Processing Latency (µs/point)")
    axes[2].set_ylabel("Latency (µs) - Log Scale")
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo18_vs_oracles_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved summary comparison plot: {plot_path}")

if __name__ == "__main__":
    main()
