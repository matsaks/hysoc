"""
Demo 30: HYSOC-G Full Evaluation — Geometric Pipelines Only

Runs three geometric compression pipelines on every CSV file in the input
directory and produces an evaluation-contract bundle compatible with Demo 26.

Pipelines:
  1. Plain DP    — Douglas-Peucker over the raw trajectory (no segmentation)
  2. Oracle-G    — STSS (offline DBSCAN-style stop detection) + DP per move
  3. HYSOC-G     — STEP (online stay-point detection) + SQUISH + DP per move

No OSM graph download, no map matching, no per-file plots.
Self-contained against src/ and scripts/evaluation_contract.py only — no
cross-imports from other demo scripts.
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
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, current_dir)  # for evaluation_contract

# ---------------------------------------------------------------------------
# src/ imports
# ---------------------------------------------------------------------------

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from core.compression import CompressionStrategy, HYSOCConfig
from core.segment import Move, Stop
from core.stream import TrajectoryStream
from engines.dp import DouglasPeuckerCompressor
from engines.squish import SquishCompressor
from engines.stop_compressor import CompressedStop, StopCompressor
from eval.sed import calculate_sed_stats
from hysoc.hysocG import HYSOCCompressor
from oracle.oracleG import OracleG

# scripts/ import (library module — not a demo)
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = os.path.join("data", "raw", "NYC_Top_1000_Longest")
DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_30_hysoc_g_full_eval")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS


# ---------------------------------------------------------------------------
# Private helpers (inlined from src/ primitives — no demo_23 import)
# ---------------------------------------------------------------------------


def _load_trajectory(filepath: str, obj_id: str) -> List:
    """Load a CSV trajectory file using TrajectoryStream."""
    stream = TrajectoryStream(
        filepath=filepath,
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
        default_obj_id=obj_id,
    )
    return list(stream.stream())


def _reconstruct_for_sed(items: list) -> tuple:
    """
    Flatten a heterogeneous item list (CompressedStop | Move) into a
    chronological point sequence suitable for SED computation.

    Returns (sed_stream, stored_points_count).
    """
    sed_stream = []
    stored_points = 0
    for item in items:
        if isinstance(item, CompressedStop):
            # Represent the stop interval by its start and end endpoints
            p_start = replace(item.centroid, timestamp=item.start_time)
            p_end = replace(item.centroid, timestamp=item.end_time)
            sed_stream.extend([p_start, p_end])
            stored_points += 1  # centroid counts as one stored point
        elif isinstance(item, Move):
            sed_stream.extend(item.points)
            stored_points += len(item.points)
    return sed_stream, stored_points


def _summarize_sed_and_cr(
    original: list, reconstructed: list, stored_points: int, latency_us: float
) -> Dict[str, Any]:
    """Compute CR + SED statistics into the canonical metric dict."""
    cr = len(original) / max(1, stored_points)
    stats = calculate_sed_stats(original, reconstructed)
    sed_errors = stats.get("sed_errors", [])
    if not sed_errors:
        avg_sed = p95_sed = max_sed = 0.0
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
        "latency_us_per_point": latency_us,
    }


def _metrics_segmented(
    original: list, items: list, latency_us: float
) -> Dict[str, Any]:
    """Metrics for pipelines that emit a list of CompressedStop / Move items."""
    sed_stream, stored = _reconstruct_for_sed(items)
    return _summarize_sed_and_cr(original, sed_stream, stored, latency_us)


def _metrics_hysoc(original: list, traj_result, latency_us: float) -> Dict[str, Any]:
    """Metrics for HYSOC-G which returns a TrajectoryResult."""
    return _summarize_sed_and_cr(
        original,
        traj_result.keypoints,
        len(traj_result.keypoints),
        latency_us,
    )


# ---------------------------------------------------------------------------
# Absolute path helper
# ---------------------------------------------------------------------------


def _to_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


# ---------------------------------------------------------------------------
# Summary boxplot
# ---------------------------------------------------------------------------


def _save_summary_plot(results: List[Dict[str, Any]], out_dir: str) -> str:
    """Produce the 1×3 boxplot figure and save it; returns the output path."""
    labels = ["Plain DP", "Oracle-G", "HYSOC-G"]
    colors = ["#bdbdbd", "#64b5f6", "#1565c0"]

    cr_data = [
        [r["plain_cr"] for r in results],
        [r["oracle_g_cr"] for r in results],
        [r["hysoc_g_cr"] for r in results],
    ]
    sed_data = [
        [r["plain_avg_sed_m"] for r in results],
        [r["oracle_g_avg_sed_m"] for r in results],
        [r["hysoc_g_avg_sed_m"] for r in results],
    ]
    lat_data = [
        [r["plain_latency_us_per_point"] for r in results],
        [r["oracle_g_latency_us_per_point"] for r in results],
        [r["hysoc_g_latency_us_per_point"] for r in results],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, data, title, ylabel, log_scale in [
        (axes[0], cr_data, "Compression Ratio (CR)", "Raw / Compressed", False),
        (axes[1], sed_data, "Information Loss (Mean SED)", "SED (m)", False),
        (axes[2], lat_data, "Processing Latency (µs/point)", "µs (log scale)", True),
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
        f"Demo 30: HYSOC-G Geometric Evaluation — {len(results)} trajectories",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo30_pipelines_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 30: HYSOC-G geometric-only full evaluation."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-trajectory CSV files.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run outputs (a timestamped subdirectory is created).",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=DEFAULT_BUFFER_CAPACITY,
        help="SQUISH buffer capacity for HYSOC-G.",
    )
    parser.add_argument(
        "--dp-epsilon-meters",
        type=float,
        default=DEFAULT_DP_EPSILON_METERS,
        help="Douglas-Peucker epsilon (metres) for all DP steps.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, process only the first N files (smoke-test mode).",
    )
    args = parser.parse_args()

    input_dir = _to_abs_path(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    csv_files.sort(
        key=lambda name: int(os.path.splitext(name)[0])
        if os.path.splitext(name)[0].isdigit()
        else 0
    )
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Demo 30: HYSOC-G Full Evaluation")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Files  : {len(csv_files)}")
    print(f"  Buffer : {args.buffer_capacity}  eps={args.dp_epsilon_meters} m")
    print()

    # --- Shared compressor instances ---
    stop_compressor = StopCompressor()
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    oracle_g = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    hysoc_config = HYSOCConfig(
        move_compression_strategy=CompressionStrategy.GEOMETRIC,
        stop_max_eps_meters=STOP_MAX_EPS_METERS,
        stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        squish_buffer_capacity=args.buffer_capacity,
        dp_epsilon_meters=args.dp_epsilon_meters,
    )

    results: List[Dict[str, Any]] = []
    contract_per_file: List[Dict[str, Any]] = []
    online_processing_time_s = 0.0

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(input_dir, fname)

        raw_points = _load_trajectory(path, obj_id)
        if len(raw_points) < 2:
            print(f"[{file_idx}/{len(csv_files)}] {obj_id}: skipped (< 2 points)")
            continue

        n_raw = len(raw_points)
        print(f"[{file_idx}/{len(csv_files)}] {obj_id}: {n_raw} raw points", end="  ")

        # ----------------------------------------------------------------
        # Pipeline 1 — Plain DP
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        plain_dp_pts = dp_compressor.compress(raw_points)
        processed_plain = [Move(points=plain_dp_pts)]
        t1 = time.perf_counter()
        dt_plain = t1 - t0
        online_processing_time_s += dt_plain
        metrics_plain = _metrics_segmented(
            raw_points, processed_plain, (dt_plain * 1e6) / n_raw
        )

        # ----------------------------------------------------------------
        # Pipeline 2 — Oracle-G (STSS + DP)
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        segments_stss = oracle_g.process(raw_points)
        processed_oracle_g = []
        for seg in segments_stss:
            if isinstance(seg, Stop):
                processed_oracle_g.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                processed_oracle_g.append(Move(points=dp_compressor.compress(seg.points)))
        t1 = time.perf_counter()
        dt_oracle_g = t1 - t0
        online_processing_time_s += dt_oracle_g
        metrics_oracle_g = _metrics_segmented(
            raw_points, processed_oracle_g, (dt_oracle_g * 1e6) / n_raw
        )
        oracle_g_n_stops = sum(1 for x in processed_oracle_g if isinstance(x, CompressedStop))
        oracle_g_n_moves = sum(1 for x in processed_oracle_g if isinstance(x, Move))

        # ----------------------------------------------------------------
        # Pipeline 3 — HYSOC-G (STEP + SQUISH + DP)
        # ----------------------------------------------------------------
        compressor_g = HYSOCCompressor(config=hysoc_config)
        t0 = time.perf_counter()
        compressed_g = compressor_g.compress(raw_points)
        t1 = time.perf_counter()
        dt_hysoc_g = t1 - t0
        online_processing_time_s += dt_hysoc_g
        metrics_hysoc_g = _metrics_hysoc(
            raw_points, compressed_g, (dt_hysoc_g * 1e6) / n_raw
        )
        hysoc_g_n_stops = len(compressed_g.stops())
        hysoc_g_n_moves = len(compressed_g.moves())

        # ----------------------------------------------------------------
        # Console summary
        # ----------------------------------------------------------------
        print(
            f"Plain={metrics_plain['cr']:.2f}×/{metrics_plain['avg_sed_m']:.2f}m  "
            f"OracleG={metrics_oracle_g['cr']:.2f}×/{metrics_oracle_g['avg_sed_m']:.2f}m  "
            f"HYSOC-G={metrics_hysoc_g['cr']:.2f}×/{metrics_hysoc_g['avg_sed_m']:.2f}m"
        )

        # ----------------------------------------------------------------
        # Flat record for CSV + per-file metrics.json
        # ----------------------------------------------------------------
        rec: Dict[str, Any] = {
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            **{f"plain_{k}": v for k, v in metrics_plain.items()},
            **{f"oracle_g_{k}": v for k, v in metrics_oracle_g.items()},
            "oracle_g_n_stops": oracle_g_n_stops,
            "oracle_g_n_moves": oracle_g_n_moves,
            **{f"hysoc_g_{k}": v for k, v in metrics_hysoc_g.items()},
            "hysoc_g_n_stops": hysoc_g_n_stops,
            "hysoc_g_n_moves": hysoc_g_n_moves,
        }
        results.append(rec)

        contract_per_file.append(
            {
                "obj_id": obj_id,
                "n_raw_points": n_raw,
                "pipelines": {
                    "plain_dp": normalize_pipeline_metrics(metrics_plain),
                    "oracle_g": normalize_pipeline_metrics(metrics_oracle_g),
                    "hysoc_g": normalize_pipeline_metrics(metrics_hysoc_g),
                },
            }
        )

        # Per-trajectory metrics.json
        obj_dir = os.path.join(out_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
            json.dump(rec, f, indent=2)

    # ----------------------------------------------------------------
    # Aggregate CSV
    # ----------------------------------------------------------------
    if not results:
        print("No trajectories processed — exiting.")
        return

    csv_path = os.path.join(out_dir, "demo30_evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved aggregate CSV      : {csv_path}")

    # ----------------------------------------------------------------
    # agg_summary.json
    # ----------------------------------------------------------------
    agg_keys = [
        "plain_cr", "oracle_g_cr", "hysoc_g_cr",
        "plain_avg_sed_m", "oracle_g_avg_sed_m", "hysoc_g_avg_sed_m",
        "plain_p95_sed_m", "oracle_g_p95_sed_m", "hysoc_g_p95_sed_m",
        "plain_max_sed_m", "oracle_g_max_sed_m", "hysoc_g_max_sed_m",
        "plain_latency_us_per_point", "oracle_g_latency_us_per_point",
        "hysoc_g_latency_us_per_point",
        "oracle_g_n_stops", "oracle_g_n_moves",
        "hysoc_g_n_stops", "hysoc_g_n_moves",
    ]
    agg_metrics: Dict[str, Any] = {"mean": {}, "median": {}}
    for key in agg_keys:
        vals = [float(r[key]) for r in results if key in r]
        agg_metrics["mean"][key] = float(np.mean(vals)) if vals else float("nan")
        agg_metrics["median"][key] = float(np.median(vals)) if vals else float("nan")

    agg_metrics["timing"] = {
        "provisioning_time_s": 0.0,
        "online_processing_time_s": online_processing_time_s,
        "end_to_end_time_s": online_processing_time_s,
        "latency_policy": "online_primary_with_end_to_end_secondary",
    }

    agg_path = os.path.join(out_dir, "agg_summary.json")
    with open(agg_path, "w", newline="") as f:
        json.dump(agg_metrics, f, indent=2)
    print(f"Saved aggregated summary : {agg_path}")

    # ----------------------------------------------------------------
    # Evaluation contract bundle
    # ----------------------------------------------------------------
    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_30_hysoc_g_full_eval",
        run_config={
            "input_dir": input_dir,
            "output_root": args.output_root,
            "buffer_capacity": args.buffer_capacity,
            "dp_epsilon_meters": args.dp_epsilon_meters,
            "max_files": args.max_files,
            "n_input_files": len(csv_files),
            "n_processed_files": len(results),
            "online_processing_time_s": online_processing_time_s,
            "latency_policy": "online_primary_with_end_to_end_secondary",
        },
        per_file_records=contract_per_file,
        metadata={
            "notes": (
                "Geometric-only evaluation: Plain DP, Oracle-G (STSS+DP), "
                "HYSOC-G (STEP+SQUISH+DP). No OSM graph download, no map matching."
            ),
        },
    )
    print(f"Saved contract bundle    : {contract_paths['contract_agg_summary']}")

    # ----------------------------------------------------------------
    # Summary boxplot
    # ----------------------------------------------------------------
    plot_path = _save_summary_plot(results, out_dir)
    print(f"Saved comparison plot    : {plot_path}")

    print("\n" + "=" * 60)
    print("Demo 30 completed.")
    print(f"Results directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
