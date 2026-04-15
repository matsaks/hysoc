# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from core.point import Point
from core.segment import Move, Stop
from eval import calculate_sed_stats
from engines.move_compression.hybrid_squish_dp import (
    HybridSquishDPCompressor,
    HybridSquishDPConfig,
)
from engines.move_compression.squish import SquishCompressor
from engines.stop_compression.compressor import CompressedStop, StopCompressor
from oracle.dpOracle import DPOracle
from oracle.oracleG import STSSOracleSklearn


DEFAULT_SUBSET_DIR = os.path.join("data", "raw", "subset_50")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS
DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_15_hybrid_squish_dp")


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
            # For SED, represent the stop duration with a start and end point.
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


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def run_one_trajectory(
    trajectory: List[Point],
    *,
    buffer_capacity: int,
    dp_epsilon_meters: float,
    stss_oracle: STSSOracleSklearn,
    stop_compressor: StopCompressor,
    squish: SquishCompressor,
    hybrid: HybridSquishDPCompressor,
    dp_oracle: DPOracle,
) -> Dict[str, object]:
    segments = stss_oracle.process(trajectory)

    processed_squish: List[object] = []
    processed_hybrid: List[object] = []
    processed_dp: List[object] = []

    short_moves = 0
    long_moves = 0

    for seg in segments:
        if isinstance(seg, Stop):
            stop_item = stop_compressor.compress(seg.points)
            processed_squish.append(stop_item)
            processed_hybrid.append(stop_item)
            processed_dp.append(stop_item)
            continue

        if isinstance(seg, Move):
            move_points = seg.points
            if len(move_points) <= buffer_capacity:
                short_moves += 1
            else:
                long_moves += 1

            squish_points = squish.compress(move_points, capacity=buffer_capacity)
            hybrid_points = hybrid.compress(
                move_points,
                capacity=buffer_capacity,
                dp_epsilon_meters=dp_epsilon_meters,
            )
            dp_points = dp_oracle.process(seg)

            processed_squish.append(Move(points=squish_points))
            processed_hybrid.append(Move(points=hybrid_points))
            processed_dp.append(Move(points=dp_points))

    # SED + CR for segmented experiments.
    sed_squish, stored_squish = reconstruct_for_sed(processed_squish)
    sed_hybrid, stored_hybrid = reconstruct_for_sed(processed_hybrid)
    sed_dp, stored_dp = reconstruct_for_sed(processed_dp)

    cr_squish = len(trajectory) / max(1, stored_squish)
    cr_hybrid = len(trajectory) / max(1, stored_hybrid)
    cr_dp = len(trajectory) / max(1, stored_dp)

    sed_stats_squish = calculate_sed_stats(trajectory, sed_squish)
    sed_stats_hybrid = calculate_sed_stats(trajectory, sed_hybrid)
    sed_stats_dp = calculate_sed_stats(trajectory, sed_dp)

    # Full-run SQUISH (no stop segmentation).
    squish_full_points = squish.compress(trajectory, capacity=buffer_capacity)
    full_run_cr_squish = len(trajectory) / max(1, len(squish_full_points))
    full_run_sed_stats_squish = calculate_sed_stats(trajectory, squish_full_points)

    # Extra bookkeeping: equality check between SQUISH and Hybrid output on the segmented stream.
    # This helps explain “no visible difference” cases.
    # We compare by length and point-by-point equality (Point is a frozen dataclass).
    squish_items = processed_squish
    hybrid_items = processed_hybrid
    identical_stream = len(squish_items) == len(hybrid_items) and all(
        str(a) == str(b) for a, b in zip(squish_items, hybrid_items)
    )

    return {
        "n_raw_points": len(trajectory),
        "n_segments": len(segments),
        "n_short_moves": short_moves,
        "n_long_moves": long_moves,
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
        "dp_oracle": {
            "compression_ratio": cr_dp,
            "stored_points": stored_dp,
            "average_sed_m": sed_stats_dp["average_sed"],
            "max_sed_m": sed_stats_dp["max_sed"],
            "rmse_m": sed_stats_dp["rmse"],
        },
        "full_run_squish": {
            "compression_ratio": full_run_cr_squish,
            "stored_points": len(squish_full_points),
            "average_sed_m": full_run_sed_stats_squish["average_sed"],
            "max_sed_m": full_run_sed_stats_squish["max_sed"],
            "rmse_m": full_run_sed_stats_squish["rmse"],
        },
        "squish_hybrid_identical_on_stream": identical_stream,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 15: Hybrid SQUISH+DP on all subset_50 trajectories.")
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--dp-refine-when-evictions", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only process the first N files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, DEFAULT_SUBSET_DIR)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"subset dir not found: {subset_dir}")

    csv_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    # Sort numerically if possible (filenames are obj_id.csv in this dataset).
    def obj_id_from_name(name: str) -> int:
        base = os.path.splitext(name)[0]
        try:
            return int(base)
        except ValueError:
            return 0

    csv_files.sort(key=obj_id_from_name)
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {len(csv_files)} trajectories in {subset_dir}")

    stss_oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    hybrid = HybridSquishDPCompressor(
        HybridSquishDPConfig(
            capacity=args.buffer_capacity,
            dp_epsilon_meters=args.dp_epsilon_meters,
            dp_refine_when_evictions=args.dp_refine_when_evictions,
        )
    )
    dp_oracle = DPOracle(epsilon_meters=args.dp_epsilon_meters)

    per_obj: Dict[str, Dict[str, object]] = {}

    for fname in csv_files:
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(subset_dir, fname)
        print(f"- {obj_id}...")
        trajectory = load_trajectory(path, obj_id=obj_id)
        if len(trajectory) < 2:
            print(f"  skipped (not enough points: {len(trajectory)})")
            continue

        metrics = run_one_trajectory(
            trajectory,
            buffer_capacity=args.buffer_capacity,
            dp_epsilon_meters=args.dp_epsilon_meters,
            stss_oracle=stss_oracle,
            stop_compressor=stop_compressor,
            squish=squish,
            hybrid=hybrid,
            dp_oracle=dp_oracle,
        )
        per_obj[obj_id] = metrics

    # Save per-object JSON.
    per_obj_path = os.path.join(out_dir, "per_object_metrics.json")
    with open(per_obj_path, "w", newline="") as f:
        json.dump(per_obj, f, indent=2)
    print(f"Saved per-object metrics: {per_obj_path}")

    # Save flat CSV for easier plotting in thesis notebooks.
    per_obj_csv_path = os.path.join(out_dir, "per_object_metrics.csv")
    with open(per_obj_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "obj_id",
                "n_raw_points",
                "n_segments",
                "n_short_moves",
                "n_long_moves",
                "squish_cr",
                "squish_avg_sed_m",
                "hybrid_cr",
                "hybrid_avg_sed_m",
                "dp_cr",
                "dp_avg_sed_m",
                "full_run_squish_cr",
                "full_run_squish_avg_sed_m",
                "squish_hybrid_identical_on_stream",
            ]
        )

        for obj_id, m in per_obj.items():
            writer.writerow(
                [
                    obj_id,
                    m["n_raw_points"],
                    m["n_segments"],
                    m["n_short_moves"],
                    m["n_long_moves"],
                    m["squish"]["compression_ratio"],
                    m["squish"]["average_sed_m"],
                    m["hybrid"]["compression_ratio"],
                    m["hybrid"]["average_sed_m"],
                    m["dp_oracle"]["compression_ratio"],
                    m["dp_oracle"]["average_sed_m"],
                    m["full_run_squish"]["compression_ratio"],
                    m["full_run_squish"]["average_sed_m"],
                    m["squish_hybrid_identical_on_stream"],
                ]
            )
    print(f"Saved per-object CSV: {per_obj_csv_path}")

    # Summary stats.
    squish_avg_sed = [m["squish"]["average_sed_m"] for m in per_obj.values()]
    hybrid_avg_sed = [m["hybrid"]["average_sed_m"] for m in per_obj.values()]
    dp_avg_sed = [m["dp_oracle"]["average_sed_m"] for m in per_obj.values()]

    squish_cr = [m["squish"]["compression_ratio"] for m in per_obj.values()]
    hybrid_cr = [m["hybrid"]["compression_ratio"] for m in per_obj.values()]
    dp_cr = [m["dp_oracle"]["compression_ratio"] for m in per_obj.values()]

    full_run_avg_sed = [m["full_run_squish"]["average_sed_m"] for m in per_obj.values()]

    summary: Dict[str, object] = {
        "n_processed": len(per_obj),
        "buffer_capacity": args.buffer_capacity,
        "dp_epsilon_meters": args.dp_epsilon_meters,
        "dp_refine_when_evictions": args.dp_refine_when_evictions,
        "mean_avg_sed_m": {
            "squish": safe_mean(squish_avg_sed),
            "hybrid": safe_mean(hybrid_avg_sed),
            "dp_oracle": safe_mean(dp_avg_sed),
            "full_run_squish": safe_mean(full_run_avg_sed),
        },
        "mean_cr": {
            "squish": safe_mean(squish_cr),
            "hybrid": safe_mean(hybrid_cr),
            "dp_oracle": safe_mean(dp_cr),
        },
        "hybrid_better_than_squish_avg_sed_count": int(sum(h < s for h, s in zip(hybrid_avg_sed, squish_avg_sed))),
        "squish_hybrid_identical_on_stream_count": int(
            sum(bool(m["squish_hybrid_identical_on_stream"]) for m in per_obj.values())
        ),
        "p95_avg_sed_m": {
            "squish": float(np.percentile(np.asarray(squish_avg_sed, dtype=float), 95)) if squish_avg_sed else 0.0,
            "hybrid": float(np.percentile(np.asarray(hybrid_avg_sed, dtype=float), 95)) if hybrid_avg_sed else 0.0,
            "dp_oracle": float(np.percentile(np.asarray(dp_avg_sed, dtype=float), 95)) if dp_avg_sed else 0.0,
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", newline="") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Plot summary comparisons (no GIS needed).
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].boxplot(
            [squish_avg_sed, hybrid_avg_sed, dp_avg_sed, full_run_avg_sed],
            tick_labels=["SQUISH (seg)", "HYBRID (seg)", "DP Oracle", "SQUISH (full)"],
            showmeans=True,
        )
        axes[0].set_title("Average SED distribution")
        axes[0].set_ylabel("Average SED (m)")
        axes[0].grid(True, alpha=0.2)

        axes[1].boxplot(
            [squish_cr, hybrid_cr, dp_cr],
            tick_labels=["SQUISH", "HYBRID", "DP Oracle"],
            showmeans=True,
        )
        axes[1].set_title("Compression ratio distribution")
        axes[1].set_ylabel("CR (|orig| / |comp|)")
        axes[1].grid(True, alpha=0.2)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, "comparison_summary.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison plot: {plot_path}")
    except Exception as e:
        print(f"Plotting skipped (matplotlib missing or failed): {e}")


if __name__ == "__main__":
    main()

