"""
Demo 32: Stop Compression Strategy Evaluation

Evaluates four stop compression strategies on isolated stop segments.
"""

# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, current_dir)

from constants.segmentation_defaults import STOP_MAX_EPS_METERS, STOP_MIN_DURATION_SECONDS
from constants.stop_compression_defaults import StopCompressionStrategy
from core.point import Point
from core.segment import Stop
from core.stream import TrajectoryStream
from engines.step import STEPSegmenter
from engines.stop_compressor import StopCompressor
from eval.sed import calculate_sed_error
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

DEFAULT_INPUT_DIR = os.path.join("data", "raw", "NYC_Top_1000_Longest")
DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_32_stop_compression_strategies")

def _load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    stream = TrajectoryStream(
        filepath=filepath,
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
        default_obj_id=obj_id,
    )
    return list(stream.stream())

def _to_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)

def _save_summary_plot(results: List[Dict[str, Any]], out_dir: str) -> str:
    strategies = ["centroid", "medoid", "snap_to_nearest", "first_point"]
    labels = ["Centroid", "Medoid", "Snap", "First"]
    colors = ["#bdbdbd", "#64b5f6", "#1565c0", "#ffb74d"]

    sed_data = [[r[f"{s}_avg_sed_m"] for r in results if not np.isnan(r.get(f"{s}_avg_sed_m", float('nan')))] for s in strategies]
    lat_data = [[r[f"{s}_latency_us_per_stop"] for r in results if not np.isnan(r.get(f"{s}_latency_us_per_stop", float('nan')))] for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, ylabel, log_scale in [
        (axes[0], sed_data, "Information Loss on Stops (Mean SED)", "SED (m)", False),
        (axes[1], lat_data, "Processing Latency (µs/stop)", "µs (log scale)", True),
    ]:
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            showmeans=True,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    fig.suptitle(
        f"Demo 32: Stop Compression Strategies — {len(results)} trajectories",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo32_stop_strategies_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 32: Stop Compression Strategy Evaluation.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    input_dir = _to_abs_path(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    csv_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Demo 32: Stop Compression Strategy Evaluation")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Files  : {len(csv_files)}")

    strategies = [
        ("centroid", StopCompressionStrategy.CENTROID),
        ("medoid", StopCompressionStrategy.MEDOID),
        ("snap_to_nearest", StopCompressionStrategy.SNAP_TO_NEAREST),
        ("first_point", StopCompressionStrategy.FIRST_POINT)
    ]

    compressors = {name: StopCompressor(strategy=strat) for name, strat in strategies}

    results: List[Dict[str, Any]] = []
    contract_per_file: List[Dict[str, Any]] = []
    online_processing_time_s = 0.0

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(input_dir, fname)

        raw_points = _load_trajectory(path, obj_id)
        if len(raw_points) < 2:
            continue

        # Extract stop segments
        segmenter = STEPSegmenter(max_eps=STOP_MAX_EPS_METERS, min_duration_seconds=STOP_MIN_DURATION_SECONDS)
        segments = segmenter.process(raw_points)
        stops = [s for s in segments if isinstance(s, Stop)]

        n_stops = len(stops)
        n_stop_points = sum(len(s.points) for s in stops)

        print(f"[{file_idx}/{len(csv_files)}] {obj_id}: {n_stops} stops ({n_stop_points} points)", end="  ")

        rec: Dict[str, Any] = {
            "obj_id": obj_id,
            "n_stops": n_stops,
            "n_stop_points": n_stop_points,
        }
        
        contract_pipelines: Dict[str, Any] = {}

        if n_stops == 0:
            print("skipped (no stops)")
            continue

        for name, _ in strategies:
            compressor = compressors[name]
            
            t0 = time.perf_counter()
            c_stops = []
            for stop in stops:
                c_stops.append(compressor.compress(stop.points))
            t1 = time.perf_counter()
            
            dt = t1 - t0
            online_processing_time_s += dt
            latency_us_per_stop = (dt * 1e6) / n_stops

            # compute SED
            all_sed = []
            for stop, c_stop in zip(stops, c_stops):
                p_start = replace(c_stop.centroid, timestamp=c_stop.start_time)
                p_end = replace(c_stop.centroid, timestamp=c_stop.end_time)
                for p_orig in stop.points:
                    all_sed.append(calculate_sed_error(p_orig, p_start, p_end))

            arr = np.asarray(all_sed, dtype=float)
            avg_sed = float(arr.mean()) if len(arr) > 0 else 0.0
            p95_sed = float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0
            max_sed = float(arr.max()) if len(arr) > 0 else 0.0

            metrics = {
                "cr": float("nan"), 
                "stored_points": n_stops,
                "avg_sed_m": avg_sed,
                "p95_sed_m": p95_sed,
                "max_sed_m": max_sed,
                "latency_us_per_stop": latency_us_per_stop,
                "latency_us_per_point": latency_us_per_stop / (n_stop_points / n_stops) if n_stops > 0 else 0.0 
            }

            rec.update({
                f"{name}_avg_sed_m": avg_sed,
                f"{name}_p95_sed_m": p95_sed,
                f"{name}_max_sed_m": max_sed,
                f"{name}_latency_us_per_stop": latency_us_per_stop
            })
            
            contract_pipelines[name] = normalize_pipeline_metrics(metrics)
            
            if name == "centroid":
                print(f"Centroid SED={avg_sed:.2f}m", end="  ")
            elif name == "snap_to_nearest":
                print(f"Snap SED={avg_sed:.2f}m", end="  ")

        print()
        results.append(rec)
        contract_per_file.append({
            "obj_id": obj_id,
            "n_raw_points": len(raw_points),
            "pipelines": contract_pipelines
        })

    if not results:
        print("No stops found in any trajectory.")
        return

    csv_path = os.path.join(out_dir, "demo32_stop_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    plot_path = _save_summary_plot(results, out_dir)
    print(f"Saved comparison plot    : {plot_path}")

    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_32_stop_compression_strategies",
        run_config={
            "input_dir": input_dir,
            "output_root": args.output_root,
            "max_files": args.max_files,
            "n_processed_files": len(results),
            "online_processing_time_s": online_processing_time_s,
        },
        per_file_records=contract_per_file,
        metadata={
            "notes": "Isolates stop segment compression strategies: Centroid, Medoid, Snap-to-Nearest, First-Point."
        }
    )
    print(f"Saved contract bundle    : {contract_paths['contract_agg_summary']}")

if __name__ == "__main__":
    main()
