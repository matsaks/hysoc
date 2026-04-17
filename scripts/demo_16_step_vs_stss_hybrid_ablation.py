# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from core.point import Point
from core.segment import Move, Stop
from eval import calculate_sed_stats
from engines.dp import DouglasPeuckerCompressor
from engines.squish import SquishCompressor
from engines.step import STEPSegmenter
from engines.stop_compressor import CompressedStop, StopCompressor
from oracle.oracleG import OracleG
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle


DEFAULT_OUTPUT_ROOT = os.path.join(
    "data", "processed", "demo_16_step_vs_stss_hybrid_ablation"
)
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS
DEFAULT_SUBSET_DIR = os.path.join("data", "raw", "subset_50")


def load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    """Load a single CSV trajectory into Points."""
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
    Reconstruct the compressed point stream used by `calculate_sed_stats`.

    Conventions (to match other demos):
    - Stop contributes two points (start_time, end_time) for SED interpolation,
      but counts as 1 "stored point" for CR.
    - Move contributes N points (as emitted by a move compressor), stored_points += N.
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


def compute_sed_bundle(original: List[Point], compressed: List[Point]) -> Dict[str, Any]:
    """
    Compute SED summaries without persisting raw per-point errors.
    """
    stats = calculate_sed_stats(original, compressed)
    sed_errors: List[float] = stats.get("sed_errors", [])
    if not sed_errors:
        return {
            "avg_sed_m": 0.0,
            "median_sed_m": 0.0,
            "p10_sed_m": 0.0,
            "p95_sed_m": 0.0,
            "p99_sed_m": 0.0,
            "max_sed_m": 0.0,
            "rmse_m": 0.0,
            "pct_sed_le_5m": 0.0,
            "pct_sed_le_10m": 0.0,
            # Keep a fixed-size vector to simplify downstream plotting.
            "sed_quantiles_p5_to_p99_m": [0.0] * 95,
        }

    arr = np.asarray(sed_errors, dtype=float)
    percentiles = list(range(5, 100))  # 5..99 inclusive => 95 values
    quantiles = np.percentile(arr, percentiles)
    pct_5 = float((arr <= 5.0).mean()) * 100.0
    pct_10 = float((arr <= 10.0).mean()) * 100.0

    return {
        "avg_sed_m": float(stats["average_sed"]),
        "median_sed_m": float(np.percentile(arr, 50)),
        "p10_sed_m": float(np.percentile(arr, 10)),
        "p95_sed_m": float(np.percentile(arr, 95)),
        "p99_sed_m": float(np.percentile(arr, 99)),
        "max_sed_m": float(stats["max_sed"]),
        "rmse_m": float(stats["rmse"]),
        "pct_sed_le_5m": pct_5,
        "pct_sed_le_10m": pct_10,
        "sed_quantiles_p5_to_p99_m": [float(x) for x in quantiles],
    }


def compute_full_run_metrics(raw_points: List[Point], compressed_points: List[Point]) -> Dict[str, Any]:
    """Compute CR + SED bundle for full-trajectory point stream variants."""
    stored_points = len(compressed_points)
    cr = len(raw_points) / max(1, stored_points)
    sed_bundle = compute_sed_bundle(raw_points, compressed_points)
    return {
        "cr": cr,
        "stored_points": stored_points,
        **sed_bundle,
    }


def compute_segmented_metrics(original: List[Point], items: List[object]) -> Dict[str, Any]:
    """Compute CR + SED bundle for STOP/MOVE segmented compressors."""
    sed_stream, stored_points = reconstruct_for_sed(items)
    cr = len(original) / max(1, stored_points)
    sed_bundle = compute_sed_bundle(original, sed_stream)
    return {
        "cr": cr,
        "stored_points": stored_points,
        **sed_bundle,
    }


def obj_id_from_name(name: str) -> int:
    base = os.path.splitext(name)[0]
    try:
        return int(base)
    except ValueError:
        return 0


def segment_step(trajectory: List[Point], *, segmenter: STEPSegmenter) -> List[Any]:
    """
    True online-style segmentation: feed points one-by-one, then flush.

    Returns a list of `Segment` objects.
    """
    segments: List[Any] = []
    for p in trajectory:
        segments.extend(segmenter.process_point(p))
    segments.extend(segmenter.flush())
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 16: STEP vs STSS hybrid ablation on subset_50."
    )
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--max-files", type=int, default=0, help="If >0, only process the first N files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--topk", type=int, default=10, help="Top-K ranking outputs (improvements/degradations).")
    parser.add_argument("--seed", type=int, default=0, help="(Unused) kept for reproducibility hooks.")
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, DEFAULT_SUBSET_DIR)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"subset directory not found: {subset_dir}")

    csv_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    csv_files.sort(key=obj_id_from_name)
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {len(csv_files)} trajectories in {subset_dir} ...")

    # Shared compressors/oracles
    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)

    stss_oracle = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    step_segmenter = STEPSegmenter(
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )

    # Variant metadata + family tags
    variant_meta: Dict[str, Dict[str, str]] = {
        # Offline oracle family (batch/upper bounds)
        "offline_dp_full": {
            "family": "offline_oracles",
            "definition": "DP on the entire raw trajectory polyline (ignore STOP/MOVE).",
        },
        "offline_dp_move_only": {
            "family": "offline_oracles",
            "definition": "STSS segmentation; compress STOP with StopCompressor and MOVE with DP.",
        },
        "squish_full_then_dp_full": {
            "family": "offline_oracles",
            "definition": "SQUISH on entire raw trajectory then DP on the resulting SQUISH output (ignore STOP/MOVE).",
        },
        # Online STEP family (truly streaming, bounded buffer)
        "squish_full": {
            "family": "online_step",
            "definition": "SQUISH on entire raw trajectory polyline (ignore STOP/MOVE); represents online move compression without segmentation.",
        },
        "step_squish_moves": {
            "family": "online_step",
            "definition": "STEP segmentation; compress STOP with StopCompressor and MOVE segments with SQUISH (buffer cap).",
        },
        "step_hybrid_dp_on_short_moves": {
            "family": "online_step",
            "definition": "STEP segmentation; Hybrid rule: if MOVE_len<=cap -> DP on full MOVE, else -> SQUISH only (no DP refinement).",
        },
        "step_hybrid_squish_then_dp_long_only": {
            "family": "online_step",
            "definition": "STEP segmentation; Hybrid rule: if MOVE_len<=cap -> DP on full MOVE, else -> SQUISH then DP on SQUISH survivors.",
        },
        "step_hybrid_squish_then_dp_all_moves": {
            "family": "online_step",
            "definition": "STEP segmentation; run SQUISH on every MOVE then DP on the SQUISH output (for both short and long MOVE segments).",
        },
        # Ablation-mixed family (offline segmentation + online compressor)
        "stss_squish_moves": {
            "family": "ablation_stss",
            "definition": "STSS segmentation; compress STOP with StopCompressor and MOVE segments with SQUISH (buffer cap).",
        },
        "stss_hybrid_dp_on_short_moves": {
            "family": "ablation_stss",
            "definition": "STSS segmentation; Hybrid rule: if MOVE_len<=cap -> DP on full MOVE, else -> SQUISH only (no DP refinement).",
        },
        "stss_hybrid_squish_then_dp_long_only": {
            "family": "ablation_stss",
            "definition": "STSS segmentation; Hybrid rule: if MOVE_len<=cap -> DP on full MOVE, else -> SQUISH then DP on SQUISH survivors.",
        },
        "stss_hybrid_squish_then_dp_all_moves": {
            "family": "ablation_stss",
            "definition": "STSS segmentation; run SQUISH on every MOVE then DP on the SQUISH output.",
        },
    }

    families_order = ["offline_oracles", "online_step", "ablation_stss"]
    family_to_variants: Dict[str, List[str]] = {fam: [] for fam in families_order}
    for vk, meta in variant_meta.items():
        family_to_variants[meta["family"]].append(vk)
    # Deterministic ordering for plots/CSV columns
    for fam in family_to_variants:
        family_to_variants[fam].sort()
    variant_keys: List[str] = []
    for fam in families_order:
        variant_keys.extend(family_to_variants[fam])

    # Per-object storage
    per_object: Dict[str, Dict[str, Any]] = {}
    branch_rows: List[Dict[str, Any]] = []

    # Wide CSV header
    common_cols = [
        "obj_id",
        "n_raw_points",
        "n_segments_stss",
        "n_short_moves_stss",
        "n_long_moves_stss",
        "n_segments_step",
        "n_short_moves_step",
        "n_long_moves_step",
    ]

    sed_cols = [
        "avg_sed_m",
        "median_sed_m",
        "p10_sed_m",
        "p95_sed_m",
        "p99_sed_m",
        "max_sed_m",
        "rmse_m",
        "pct_sed_le_5m",
        "pct_sed_le_10m",
    ]

    metrics_cols: List[str] = []
    for vk in variant_keys:
        metrics_cols.extend([f"{vk}_cr", f"{vk}_stored_points"])
        for sc in sed_cols:
            metrics_cols.append(f"{vk}_{sc}")

    per_object_csv_path = os.path.join(out_dir, "per_object_metrics.csv")
    with open(per_object_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=common_cols + metrics_cols)
        writer.writeheader()

        for fname in csv_files:
            obj_id = os.path.splitext(fname)[0]
            path = os.path.join(subset_dir, fname)
            print(f"- {obj_id}...")

            raw_points = load_trajectory(path, obj_id=obj_id)
            if len(raw_points) < 2:
                print(f"  skipped (not enough points: {len(raw_points)})")
                continue

            # Full-run variants
            squish_full_points = squish.compress(raw_points, capacity=args.buffer_capacity)
            dp_full_points = dp_compressor.compress(raw_points)
            dp_on_squish_full_points = dp_compressor.compress(squish_full_points)

            metrics_full: Dict[str, Dict[str, Any]] = {
                "offline_dp_full": compute_full_run_metrics(raw_points, dp_full_points),
                "squish_full": compute_full_run_metrics(raw_points, squish_full_points),
                "squish_full_then_dp_full": compute_full_run_metrics(raw_points, dp_on_squish_full_points),
            }

            # STSS segmentation (ablation + offline move-only baseline)
            segments_stss = stss_oracle.process(raw_points)

            processed_offline_dp_move_only: List[object] = []
            processed_stss_squish_moves: List[object] = []
            processed_stss_hybrid_dp_on_short: List[object] = []
            processed_stss_hybrid_squish_then_dp_long_only: List[object] = []
            processed_stss_hybrid_squish_then_dp_all_moves: List[object] = []

            n_short_moves_stss = 0
            n_long_moves_stss = 0
            stss_dp_branch_moves = 0
            stss_squish_branch_moves = 0
            stss_refine_long_only_calls = 0
            stss_refine_all_moves_calls = 0

            for seg in segments_stss:
                if isinstance(seg, Stop):
                    stop_item = stop_compressor.compress(seg.points)
                    processed_offline_dp_move_only.append(stop_item)
                    processed_stss_squish_moves.append(stop_item)
                    processed_stss_hybrid_dp_on_short.append(stop_item)
                    processed_stss_hybrid_squish_then_dp_long_only.append(stop_item)
                    processed_stss_hybrid_squish_then_dp_all_moves.append(stop_item)
                    continue

                if isinstance(seg, Move):
                    move_points = seg.points
                    is_short = len(move_points) <= args.buffer_capacity
                    if is_short:
                        n_short_moves_stss += 1
                        stss_dp_branch_moves += 1
                    else:
                        n_long_moves_stss += 1
                        stss_squish_branch_moves += 1

                    squish_move = squish.compress(move_points, capacity=args.buffer_capacity)
                    dp_move = dp_compressor.compress(move_points)
                    dp_on_squish_move = dp_compressor.compress(squish_move)

                    # offline_dp_move_only
                    processed_offline_dp_move_only.append(Move(points=dp_move))

                    # stss_squish_moves
                    processed_stss_squish_moves.append(Move(points=squish_move))

                    # stss_hybrid_dp_on_short_moves
                    processed_stss_hybrid_dp_on_short.append(Move(points=dp_move if is_short else squish_move))

                    # stss_hybrid_squish_then_dp_long_only
                    if is_short:
                        processed_stss_hybrid_squish_then_dp_long_only.append(Move(points=dp_move))
                    else:
                        processed_stss_hybrid_squish_then_dp_long_only.append(Move(points=dp_on_squish_move))
                        stss_refine_long_only_calls += 1

                    # stss_hybrid_squish_then_dp_all_moves
                    processed_stss_hybrid_squish_then_dp_all_moves.append(Move(points=dp_on_squish_move))
                    stss_refine_all_moves_calls += 1

            metrics_segmented_stss: Dict[str, Dict[str, Any]] = {
                "offline_dp_move_only": compute_segmented_metrics(raw_points, processed_offline_dp_move_only),
                "stss_squish_moves": compute_segmented_metrics(raw_points, processed_stss_squish_moves),
                "stss_hybrid_dp_on_short_moves": compute_segmented_metrics(raw_points, processed_stss_hybrid_dp_on_short),
                "stss_hybrid_squish_then_dp_long_only": compute_segmented_metrics(
                    raw_points, processed_stss_hybrid_squish_then_dp_long_only
                ),
                "stss_hybrid_squish_then_dp_all_moves": compute_segmented_metrics(
                    raw_points, processed_stss_hybrid_squish_then_dp_all_moves
                ),
            }

            # STEP segmentation (online family)
            # NOTE: we reuse the same instance but must reset its internal state by re-instantiating.
            step_segmenter_local = STEPSegmenter(
                max_eps=STOP_MAX_EPS_METERS,
                min_duration_seconds=STOP_MIN_DURATION_SECONDS,
            )
            segments_step = segment_step(raw_points, segmenter=step_segmenter_local)

            processed_step_squish_moves: List[object] = []
            processed_step_hybrid_dp_on_short: List[object] = []
            processed_step_hybrid_squish_then_dp_long_only: List[object] = []
            processed_step_hybrid_squish_then_dp_all_moves: List[object] = []

            n_short_moves_step = 0
            n_long_moves_step = 0
            step_dp_branch_moves = 0
            step_squish_branch_moves = 0
            step_refine_long_only_calls = 0
            step_refine_all_moves_calls = 0

            for seg in segments_step:
                if isinstance(seg, Stop):
                    stop_item = stop_compressor.compress(seg.points)
                    processed_step_squish_moves.append(stop_item)
                    processed_step_hybrid_dp_on_short.append(stop_item)
                    processed_step_hybrid_squish_then_dp_long_only.append(stop_item)
                    processed_step_hybrid_squish_then_dp_all_moves.append(stop_item)
                    continue

                if isinstance(seg, Move):
                    move_points = seg.points
                    is_short = len(move_points) <= args.buffer_capacity
                    if is_short:
                        n_short_moves_step += 1
                        step_dp_branch_moves += 1
                    else:
                        n_long_moves_step += 1
                        step_squish_branch_moves += 1

                    squish_move = squish.compress(move_points, capacity=args.buffer_capacity)
                    dp_move = dp_compressor.compress(move_points)
                    dp_on_squish_move = dp_compressor.compress(squish_move)

                    # step_squish_moves
                    processed_step_squish_moves.append(Move(points=squish_move))

                    # step_hybrid_dp_on_short_moves
                    processed_step_hybrid_dp_on_short.append(Move(points=dp_move if is_short else squish_move))

                    # step_hybrid_squish_then_dp_long_only
                    if is_short:
                        processed_step_hybrid_squish_then_dp_long_only.append(Move(points=dp_move))
                    else:
                        processed_step_hybrid_squish_then_dp_long_only.append(Move(points=dp_on_squish_move))
                        step_refine_long_only_calls += 1

                    # step_hybrid_squish_then_dp_all_moves
                    processed_step_hybrid_squish_then_dp_all_moves.append(Move(points=dp_on_squish_move))
                    step_refine_all_moves_calls += 1

            metrics_segmented_step: Dict[str, Dict[str, Any]] = {
                "step_squish_moves": compute_segmented_metrics(raw_points, processed_step_squish_moves),
                "step_hybrid_dp_on_short_moves": compute_segmented_metrics(raw_points, processed_step_hybrid_dp_on_short),
                "step_hybrid_squish_then_dp_long_only": compute_segmented_metrics(
                    raw_points, processed_step_hybrid_squish_then_dp_long_only
                ),
                "step_hybrid_squish_then_dp_all_moves": compute_segmented_metrics(
                    raw_points, processed_step_hybrid_squish_then_dp_all_moves
                ),
            }

            # Combine all variants for this object
            obj_metrics: Dict[str, Any] = {}
            obj_metrics.update(metrics_full)
            obj_metrics.update(metrics_segmented_stss)
            obj_metrics.update(metrics_segmented_step)

            # Optional debug flags: check whether the hybrid DP-on-short output differs from SQUISH baseline.
            debug_step_hybrid_short_identical_on_stream = (
                processed_step_hybrid_dp_on_short == processed_step_squish_moves
            )
            debug_stss_hybrid_short_identical_on_stream = processed_stss_hybrid_dp_on_short == processed_stss_squish_moves

            per_object[obj_id] = {
                "n_raw_points": len(raw_points),
                "n_segments_stss": len(segments_stss),
                "n_short_moves_stss": n_short_moves_stss,
                "n_long_moves_stss": n_long_moves_stss,
                "n_segments_step": len(segments_step),
                "n_short_moves_step": n_short_moves_step,
                "n_long_moves_step": n_long_moves_step,
                "variants": obj_metrics,
                "debug": {
                    "step_hybrid_dp_on_short_identical_to_step_squish_moves": bool(debug_step_hybrid_short_identical_on_stream),
                    "stss_hybrid_dp_on_short_identical_to_stss_squish_moves": bool(debug_stss_hybrid_short_identical_on_stream),
                },
            }

            # Branch decision counts output row
            branch_rows.append(
                {
                    "obj_id": obj_id,
                    "n_raw_points": len(raw_points),
                    "n_segments_stss": len(segments_stss),
                    "n_short_moves_stss": n_short_moves_stss,
                    "n_long_moves_stss": n_long_moves_stss,
                    "stss_hybrid_dp_on_short_dp_branch_moves": stss_dp_branch_moves,
                    "stss_hybrid_dp_on_short_squish_branch_moves": stss_squish_branch_moves,
                    "stss_hybrid_squish_then_dp_long_only_refine_calls": stss_refine_long_only_calls,
                    "stss_hybrid_squish_then_dp_all_moves_refine_calls": stss_refine_all_moves_calls,
                    "n_segments_step": len(segments_step),
                    "n_short_moves_step": n_short_moves_step,
                    "n_long_moves_step": n_long_moves_step,
                    "step_hybrid_dp_on_short_dp_branch_moves": step_dp_branch_moves,
                    "step_hybrid_dp_on_short_squish_branch_moves": step_squish_branch_moves,
                    "step_hybrid_squish_then_dp_long_only_refine_calls": step_refine_long_only_calls,
                    "step_hybrid_squish_then_dp_all_moves_refine_calls": step_refine_all_moves_calls,
                }
            )

            # Write wide CSV row
            row: Dict[str, Any] = {
                "obj_id": obj_id,
                "n_raw_points": len(raw_points),
                "n_segments_stss": len(segments_stss),
                "n_short_moves_stss": n_short_moves_stss,
                "n_long_moves_stss": n_long_moves_stss,
                "n_segments_step": len(segments_step),
                "n_short_moves_step": n_short_moves_step,
                "n_long_moves_step": n_long_moves_step,
            }
            for vk in variant_keys:
                m = obj_metrics[vk]
                row[f"{vk}_cr"] = m["cr"]
                row[f"{vk}_stored_points"] = m["stored_points"]
                for sc in sed_cols:
                    row[f"{vk}_{sc}"] = m[sc]

            writer.writerow(row)

    # Write JSON (full quantile list stored inside per-object metrics)
    per_object_json_path = os.path.join(out_dir, "per_object_metrics.json")
    with open(per_object_json_path, "w", newline="") as f:
        json.dump(per_object, f, indent=2)
    print(f"Saved per-object JSON: {per_object_json_path}")

    # Variant definitions
    variant_def_path = os.path.join(out_dir, "variant_definitions.json")
    with open(variant_def_path, "w", newline="") as f:
        json.dump(variant_meta, f, indent=2)
    print(f"Saved variant definitions: {variant_def_path}")

    # Branch decision counts
    branch_path = os.path.join(out_dir, "branch_decisions_counts.csv")
    if branch_rows:
        fieldnames = list(branch_rows[0].keys())
    else:
        fieldnames = []
    with open(branch_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in branch_rows:
            writer.writerow(r)
    print(f"Saved branch decisions: {branch_path}")

    # Ranked deltas: compare hybrid variants to their family baseline (SQUISH within same segmentation family).
    ranked_rows: List[Dict[str, Any]] = []

    def baseline_and_hybrids(family: str) -> Tuple[str, List[str]]:
        if family == "online_step":
            return "step_squish_moves", [
                "step_hybrid_dp_on_short_moves",
                "step_hybrid_squish_then_dp_long_only",
                "step_hybrid_squish_then_dp_all_moves",
            ]
        if family == "ablation_stss":
            return "stss_squish_moves", [
                "stss_hybrid_dp_on_short_moves",
                "stss_hybrid_squish_then_dp_long_only",
                "stss_hybrid_squish_then_dp_all_moves",
            ]
        raise ValueError(f"Unsupported family for ranking: {family}")

    k = max(1, args.topk)
    for fam in ["online_step", "ablation_stss"]:
        baseline_key, hybrid_keys = baseline_and_hybrids(fam)
        deltas: List[Tuple[str, str, float]] = []
        for hkey in hybrid_keys:
            deltas = []
            for obj_id, rec in per_object.items():
                base = rec["variants"][baseline_key]["avg_sed_m"]
                hyp = rec["variants"][hkey]["avg_sed_m"]
                deltas.append((obj_id, hkey, hyp - base))
            deltas.sort(key=lambda x: x[2])

            improvements = deltas[: min(k, len(deltas))]
            degradations = deltas[-min(k, len(deltas)) :]

            for obj_id, hkey_inner, delta in improvements:
                base_m = per_object[obj_id]["variants"][baseline_key]
                hyp_m = per_object[obj_id]["variants"][hkey_inner]
                ranked_rows.append(
                    {
                        "family": fam,
                        "comparison": f"{hkey_inner}_minus_{baseline_key}",
                        "obj_id": obj_id,
                        "mean_sed_delta_m": float(delta),
                        "baseline_avg_sed_m": float(base_m["avg_sed_m"]),
                        "hybrid_avg_sed_m": float(hyp_m["avg_sed_m"]),
                        "baseline_cr": float(base_m["cr"]),
                        "hybrid_cr": float(hyp_m["cr"]),
                        "direction": "improvement",
                    }
                )
            for obj_id, hkey_inner, delta in degradations:
                base_m = per_object[obj_id]["variants"][baseline_key]
                hyp_m = per_object[obj_id]["variants"][hkey_inner]
                ranked_rows.append(
                    {
                        "family": fam,
                        "comparison": f"{hkey_inner}_minus_{baseline_key}",
                        "obj_id": obj_id,
                        "mean_sed_delta_m": float(delta),
                        "baseline_avg_sed_m": float(base_m["avg_sed_m"]),
                        "hybrid_avg_sed_m": float(hyp_m["avg_sed_m"]),
                        "baseline_cr": float(base_m["cr"]),
                        "hybrid_cr": float(hyp_m["cr"]),
                        "direction": "degradation",
                    }
                )

    ranked_path = os.path.join(out_dir, "ranked_deltas.csv")
    with open(ranked_path, "w", newline="") as f:
        if not ranked_rows:
            f.write("")
        else:
            fieldnames = list(ranked_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ranked_rows:
                writer.writerow(r)
    print(f"Saved ranked deltas: {ranked_path}")

    # Summary aggregation across objects
    def gather_scalar(vk: str, key: str) -> List[float]:
        values: List[float] = []
        for rec in per_object.values():
            values.append(float(rec["variants"][vk][key]))
        return values

    variants_data: Dict[str, Any] = {}
    for vk in variant_keys:
        avg_sed = gather_scalar(vk, "avg_sed_m")
        crs = gather_scalar(vk, "cr")
        pct5 = gather_scalar(vk, "pct_sed_le_5m")
        pct10 = gather_scalar(vk, "pct_sed_le_10m")

        arr_avg = np.asarray(avg_sed, dtype=float)
        arr_cr = np.asarray(crs, dtype=float)
        arr_pct5 = np.asarray(pct5, dtype=float)
        arr_pct10 = np.asarray(pct10, dtype=float)

        variants_data[vk] = {
            "family": variant_meta[vk]["family"],
            "mean_avg_sed_m": float(arr_avg.mean()) if len(arr_avg) else 0.0,
            "median_avg_sed_m": float(np.percentile(arr_avg, 50)) if len(arr_avg) else 0.0,
            "p95_avg_sed_m": float(np.percentile(arr_avg, 95)) if len(arr_avg) else 0.0,
            "p99_avg_sed_m": float(np.percentile(arr_avg, 99)) if len(arr_avg) else 0.0,
            "mean_cr": float(arr_cr.mean()) if len(arr_cr) else 0.0,
            "median_cr": float(np.percentile(arr_cr, 50)) if len(arr_cr) else 0.0,
            "mean_pct_sed_le_5m": float(arr_pct5.mean()) if len(arr_pct5) else 0.0,
            "mean_pct_sed_le_10m": float(arr_pct10.mean()) if len(arr_pct10) else 0.0,
        }

    summary = {
        "n_processed": len(per_object),
        "buffer_capacity": args.buffer_capacity,
        "dp_epsilon_meters": args.dp_epsilon_meters,
        "variant_families": family_to_variants,
        "variant_definitions": variant_meta,
        "variants": variants_data,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", newline="") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    contract_per_file: List[Dict[str, Any]] = []
    for obj_id, rec in per_object.items():
        pipelines = {vk: normalize_pipeline_metrics(rec["variants"][vk]) for vk in variant_keys}
        contract_per_file.append(
            {
                "obj_id": obj_id,
                "n_raw_points": int(rec["n_raw_points"]),
                "pipelines": pipelines,
            }
        )

    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_16_step_vs_stss_hybrid_ablation",
        run_config={
            "buffer_capacity": args.buffer_capacity,
            "dp_epsilon_meters": args.dp_epsilon_meters,
            "max_files": args.max_files,
            "subset_dir": subset_dir,
            "n_input_files": len(csv_files),
            "n_processed_files": len(per_object),
            "latency_note": "Per-variant latency is not measured in this demo and is exported as null.",
        },
        per_file_records=contract_per_file,
        metadata={
            "variant_families": family_to_variants,
        },
    )
    print(f"Saved contract bundle: {contract_paths['contract_agg_summary']}")

    # Comparison plot grouped by family
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        for row, fam in enumerate(families_order):
            sed_data = [gather_scalar(vk, "avg_sed_m") for vk in family_to_variants[fam]]
            sed_labels = family_to_variants[fam]
            axes[row, 0].boxplot(sed_data, tick_labels=sed_labels, showmeans=True)
            axes[row, 0].set_title(f"{fam}: avg SED distribution")
            axes[row, 0].set_ylabel("avg SED (m)")
            axes[row, 0].grid(True, alpha=0.2)

            cr_data = [gather_scalar(vk, "cr") for vk in family_to_variants[fam]]
            cr_labels = family_to_variants[fam]
            axes[row, 1].boxplot(cr_data, tick_labels=cr_labels, showmeans=True)
            axes[row, 1].set_title(f"{fam}: CR distribution")
            axes[row, 1].set_ylabel("CR")
            axes[row, 1].grid(True, alpha=0.2)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, "comparison_summary.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison plot: {plot_path}")
    except Exception as e:
        print(f"Plotting skipped (matplotlib missing or failed): {e}")


if __name__ == "__main__":
    main()

