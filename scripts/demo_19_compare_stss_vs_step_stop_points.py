import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STSS_MAX_EPS_METERS,
    STSS_MIN_DURATION_SECONDS,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from core.point import Point
from core.segment import Move, Stop
from engines.step import STEPSegmenter
from oracle.oracleG import OracleG

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_19_compare_stss_vs_step")
DEFAULT_SUBSET_DIR = os.path.join("data", "raw", "NYC_100")

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

def count_stops_and_points(segments: List[Any]) -> Dict[str, int]:
    num_stop_segments = 0
    num_stop_points = 0
    for seg in segments:
        if isinstance(seg, Stop):
            num_stop_segments += 1
            num_stop_points += len(seg.points)
    return {
        "num_stop_segments": num_stop_segments,
        "num_stop_points": num_stop_points
    }

def plot_trajectory_comparison(out_path: str, raw_points: List[Point], stss_segments: List[Any], step_segments: List[Any], title: str):
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import Point as ShapelyPoint
        import contextily as ctx
    except ImportError:
        print("Missing plotting dependencies (geopandas, contextily, shapely). Skipping plot.")
        return

    def extract_stops(segments: List[Any]):
        stop_points = []
        stop_centroids = []
        for seg in segments:
            if isinstance(seg, Stop):
                for p in seg.points:
                    stop_points.append(ShapelyPoint(p.lon, p.lat))
                if seg.points:
                    clon = sum(p.lon for p in seg.points) / len(seg.points)
                    clat = sum(p.lat for p in seg.points) / len(seg.points)
                    stop_centroids.append(ShapelyPoint(clon, clat))
        return stop_points, stop_centroids

    raw_pts = [ShapelyPoint(p.lon, p.lat) for p in raw_points] if raw_points else []
    stss_pts, stss_centroids = extract_stops(stss_segments)
    step_pts, step_centroids = extract_stops(step_segments)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    ax_stss, ax_step = axes.ravel()

    def plot_panel(ax, stop_pts, stop_centroids, algo_name):
        # Background context (Gray)
        if raw_pts:
            gpd.GeoSeries(raw_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color="gray", markersize=5, alpha=0.3)

        if stop_pts:
            gpd.GeoSeries(stop_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color="red", markersize=15, alpha=0.8, label="Stop Points")
        
        if stop_centroids:
            gpd.GeoSeries(stop_centroids, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color="blue", markersize=150, marker="*", alpha=0.9, label="Stop Segments")
            
        ax.legend(loc="upper right")
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title(f"{algo_name} ({len(stop_centroids)} segments, {len(stop_pts)} points)", fontsize=14)

    plot_panel(ax_stss, stss_pts, stss_centroids, "STSS Oracle")
    plot_panel(ax_step, step_pts, step_centroids, "STEP Online")

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 19: STSS vs STEP comparison for Stop Points.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, DEFAULT_SUBSET_DIR)
    if not os.path.exists(subset_dir):
        print(f"Directory not found: {subset_dir}")
        return

    csv_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    csv_files.sort(key=lambda name: int(os.path.splitext(name)[0]) if os.path.splitext(name)[0].isdigit() else 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running STSS vs STEP Evaluation on {len(csv_files)} trajectories in {subset_dir} ...")

    stss_oracle = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STSS_MAX_EPS_METERS,
        min_duration_seconds=STSS_MIN_DURATION_SECONDS,
    )

    results = []
    skipped_count = 0

    for idx, fname in enumerate(csv_files):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(subset_dir, fname)
        raw_points = load_trajectory(path, obj_id=obj_id)
        n_raw = len(raw_points)

        print(f"[{idx+1}/{len(csv_files)}] Processing {fname} ({n_raw} points)...", end="")

        if n_raw < 2:
            print(" Skipped (too few points).")
            skipped_count += 1
            continue
            
        # 1. Pipeline: STSS Oracle Offline
        t0_stss = time.perf_counter()
        segments_stss = stss_oracle.process(raw_points)
        t1_stss = time.perf_counter()
        latency_stss_ms = (t1_stss - t0_stss) * 1000
        metrics_stss = count_stops_and_points(segments_stss)

        # 2. Pipeline: STEP Online Stream
        step_segmenter = STEPSegmenter(
            max_eps=STOP_MAX_EPS_METERS,
            min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        )
        t0_step = time.perf_counter()
        segments_step = []
        for p in raw_points:
            segments_step.extend(step_segmenter.process_point(p))
        segments_step.extend(step_segmenter.flush())
        t1_step = time.perf_counter()
        latency_step_ms = (t1_step - t0_step) * 1000
        metrics_step = count_stops_and_points(segments_step)

        print(f" done. STSS_stop_segments: {metrics_stss['num_stop_segments']}, STEP_stop_segments: {metrics_step['num_stop_segments']}")

        rec = {
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            "stss_num_stop_segments": metrics_stss["num_stop_segments"],
            "stss_num_stop_points": metrics_stss["num_stop_points"],
            "stss_latency_ms": latency_stss_ms,
            "step_num_stop_segments": metrics_step["num_stop_segments"],
            "step_num_stop_points": metrics_step["num_stop_points"],
            "step_latency_ms": latency_step_ms
        }
        results.append(rec)
        
        # --- PER-FILE OUTPUTS ---
        obj_dir = os.path.join(out_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        
        # Save metrics JSON
        with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
            json.dump(rec, f, indent=2)
            
        # Generate plot
        plot_trajectory_comparison(
            out_path=os.path.join(obj_dir, "stss_vs_step_plot.png"),
            raw_points=raw_points,
            stss_segments=segments_stss,
            step_segments=segments_step,
            title=f"Trajectory {obj_id} - STSS vs STEP Stops"
        )

    # Aggregate CSV
    csv_path = os.path.join(out_dir, "demo19_stss_vs_step_metrics.csv")
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved evaluation aggregate metrics: {csv_path}")

        # Summary mapping (sums and means)
        agg_metrics = {
            "meta": {
                "n_input_files": int(len(csv_files)),
                "n_processed_files": int(len(results)),
                "n_skipped_files": int(skipped_count),
            },
            "total": {},
            "mean": {},
        }
        keys_to_sum = [
            "n_raw_points", 
            "stss_num_stop_segments", "stss_num_stop_points", "stss_latency_ms",
            "step_num_stop_segments", "step_num_stop_points", "step_latency_ms"
        ]
        
        for key in keys_to_sum:
            vals = [float(r[key]) for r in results]
            agg_metrics["total"][key] = float(np.sum(vals))
            agg_metrics["mean"][key] = float(np.mean(vals))
        
        agg_path = os.path.join(out_dir, "agg_summary.json")
        with open(agg_path, "w", newline="") as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"Saved aggregated statistics: {agg_path}")

        # Generate aggregated comparison bar chart
        try:
            total_stss_stop_segments = int(agg_metrics["total"]["stss_num_stop_segments"])
            total_step_stop_segments = int(agg_metrics["total"]["step_num_stop_segments"])
            total_stss_points = int(agg_metrics["total"]["stss_num_stop_points"])
            total_step_points = int(agg_metrics["total"]["step_num_stop_points"])

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Stops bar chart
            ax[0].bar(["STSS", "STEP"], [total_stss_stop_segments, total_step_stop_segments], color=["blue", "green"])
            ax[0].set_title(
                f"Total Number of Stop Segments (Across {len(results)} files)"
            )
            ax[0].set_ylabel("Number of Stop Segments")
            for i, v in enumerate([total_stss_stop_segments, total_step_stop_segments]):
                ax[0].text(i, v + (max(total_stss_stop_segments, total_step_stop_segments)*0.01), str(v), ha='center')

            # Stop Points bar chart
            ax[1].bar(["STSS", "STEP"], [total_stss_points, total_step_points], color=["blue", "green"])
            ax[1].set_title(
                f"Total Number of Stop Points (Across {len(results)} files)"
            )
            ax[1].set_ylabel("Number of Stop Points")
            for i, v in enumerate([total_stss_points, total_step_points]):
                ax[1].text(i, v + (max(total_stss_points, total_step_points)*0.01), str(v), ha='center')

            plt.tight_layout()
            summary_plot_path = os.path.join(out_dir, "demo19_summary_plot.png")
            fig.savefig(summary_plot_path, dpi=200)
            print(f"Saved summary plot: {summary_plot_path}")
            
        except Exception as e:
            print(f"Could not generate summary chart: {e}")

if __name__ == "__main__":
    main()
