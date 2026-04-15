# ruff: noqa: E402

"""
Demo 21: HYSOC Full Evaluation — Geometric + Network-Semantic vs Oracles

Runs 5 compression pipelines on every file in data/raw/subset_50/:

  1. Plain DP              — no segmentation, whole-trajectory DP
  2. Oracle-G (STSS + DP)  — offline geometric oracle
  3. HYSOC-G (STEP + SQUISH/DP) — online geometric
  4. Oracle-N (STSS + STC) — offline network-semantic oracle
  5. HYSOC-N (STEP + TRACE)— online network-semantic

Metrics per pipeline: CR, Mean SED, P95 SED, Max SED, Latency µs/point.
Outputs: per-file metrics JSON, aggregate CSV, aggregate summary JSON,
         1×3 boxplot comparison (CR / SED / Latency).
"""

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

# Add project root to sys.path
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
from engines.move_compression.dp import DouglasPeuckerCompressor
from engines.move_compression.squish import SquishCompressor
from engines.segmentation.step import STEPSegmenter
from engines.stop_compression.compressor import CompressedStop, StopCompressor
from hysoc.hysocG import HYSOCCompressor
from core.compression import HYSOCConfig, CompressionStrategy
from engines.map_matching.matcher import OnlineMapMatcher
from oracle.oracleG import STSSOracleSklearn
from oracle.oracleN import STCOracle
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_21_hysoc_vs_oracles")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS
DEFAULT_SUBSET_DIR = os.path.join("data", "raw", "subset_50")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    """Load a trajectory CSV file into a list of Points."""
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


def download_osm_graph(raw_points: List[Point]):
    """Download an OSM driving graph covering the bounding-box of raw_points."""
    try:
        import osmnx as ox
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "osmnx is required for network-semantic pipelines. Install it with `pip install osmnx`."
        ) from exc

    lats = [p.lat for p in raw_points]
    lons = [p.lon for p in raw_points]
    north, south = max(lats) + 0.01, min(lats) - 0.01
    east, west = max(lons) + 0.01, min(lons) - 0.01
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    return G


def reconstruct_for_sed(items: List[object]) -> Tuple[List[Point], int]:
    """Flatten a list of CompressedStop / Move items into a point stream for SED."""
    sed_stream: List[Point] = []
    stored_points = 0
    for item in items:
        if isinstance(item, CompressedStop):
            p_start = Point(
                lat=item.centroid.lat, lon=item.centroid.lon,
                timestamp=item.start_time, obj_id=item.centroid.obj_id,
            )
            p_end = Point(
                lat=item.centroid.lat, lon=item.centroid.lon,
                timestamp=item.end_time, obj_id=item.centroid.obj_id,
            )
            sed_stream.extend([p_start, p_end])
            stored_points += 1
        elif isinstance(item, Move):
            sed_stream.extend(item.points)
            stored_points += len(item.points)
    return sed_stream, stored_points


def compute_segmented_metrics(
    original: List[Point], items: List[object], latency_us: float
) -> Dict[str, Any]:
    """Compute CR, SED statistics, and latency for a pipeline represented as items."""
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
        "latency_us_per_point": latency_us,
    }


def compute_hysoc_metrics(
    original: List[Point],
    compressed_trajectory,
    latency_us: float,
) -> Dict[str, Any]:
    """Compute metrics for a HYSOCCompressor result (CompressedTrajectory)."""
    reconstructed = compressed_trajectory.get_reconstructed_points()
    stored_points = compressed_trajectory.total_compressed_points
    cr = len(original) / max(1, stored_points)
    stats = calculate_sed_stats(original, reconstructed)
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
        "latency_us_per_point": latency_us,
    }


def map_match_points(raw_points: List[Point], G) -> List[Point]:
    """Map-match a full list of points using OnlineMapMatcher (batch helper)."""
    matcher = OnlineMapMatcher(G)
    matched: List[Point] = []
    for p in raw_points:
        result = matcher.process_point(p)
        if result is not None:
            matched.append(result)
    matched.extend(matcher.flush())
    return matched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 21: HYSOC (G + N) vs Oracles full evaluation."
    )
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    subset_dir = os.path.join(project_root, DEFAULT_SUBSET_DIR)
    csv_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    csv_files.sort(
        key=lambda name: int(os.path.splitext(name)[0])
        if os.path.splitext(name)[0].isdigit()
        else 0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running HYSOC Full Evaluation on {len(csv_files)} trajectories in {subset_dir} ...")
    print(f"Output directory: {out_dir}\n")

    # Shared compressors for geometric pipelines
    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    stss_oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    stc_oracle = STCOracle()

    results = []
    contract_per_file: List[Dict[str, Any]] = []

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(subset_dir, fname)
        raw_points = load_trajectory(path, obj_id=obj_id)
        if len(raw_points) < 2:
            print(f"[{file_idx}/{len(csv_files)}] {obj_id}: skipped (<2 points)")
            continue

        n_raw = len(raw_points)
        print(f"\n[{file_idx}/{len(csv_files)}] {obj_id}: {n_raw} points")

        # ==============================================================
        # 1. Plain DP (no segmentation)
        # ==============================================================
        t0 = time.perf_counter()
        plain_dp_points = dp_compressor.compress(raw_points)
        processed_plain = [Move(points=plain_dp_points)]
        t1 = time.perf_counter()
        latency_us_plain = ((t1 - t0) * 1e6) / n_raw
        metrics_plain = compute_segmented_metrics(raw_points, processed_plain, latency_us_plain)
        print(f"  Plain DP:   CR={metrics_plain['cr']:.2f}  SED={metrics_plain['avg_sed_m']:.2f}m  Lat={latency_us_plain:.1f}µs/pt")

        # ==============================================================
        # 2. Oracle-G (STSS + DP)
        # ==============================================================
        t0 = time.perf_counter()
        segments_stss = stss_oracle.process(raw_points)
        processed_oracle_g = []
        for seg in segments_stss:
            if isinstance(seg, Stop):
                processed_oracle_g.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                processed_oracle_g.append(Move(points=dp_compressor.compress(seg.points)))
        t1 = time.perf_counter()
        latency_us_oracle_g = ((t1 - t0) * 1e6) / n_raw
        metrics_oracle_g = compute_segmented_metrics(raw_points, processed_oracle_g, latency_us_oracle_g)
        print(f"  Oracle-G:   CR={metrics_oracle_g['cr']:.2f}  SED={metrics_oracle_g['avg_sed_m']:.2f}m  Lat={latency_us_oracle_g:.1f}µs/pt")

        # ==============================================================
        # 3. HYSOC-G (STEP + SQUISH/DP) — via HYSOCCompressor
        # ==============================================================
        config_g = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.GEOMETRIC,
            stop_max_eps_meters=STOP_MAX_EPS_METERS,
            stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
            squish_buffer_capacity=args.buffer_capacity,
            dp_epsilon_meters=args.dp_epsilon_meters,
        )
        compressor_g = HYSOCCompressor(config=config_g)
        t0 = time.perf_counter()
        compressed_g = compressor_g.compress(raw_points)
        t1 = time.perf_counter()
        latency_us_hysoc_g = ((t1 - t0) * 1e6) / n_raw
        metrics_hysoc_g = compute_hysoc_metrics(raw_points, compressed_g, latency_us_hysoc_g)
        print(f"  HYSOC-G:    CR={metrics_hysoc_g['cr']:.2f}  SED={metrics_hysoc_g['avg_sed_m']:.2f}m  Lat={latency_us_hysoc_g:.1f}µs/pt")

        # ==============================================================
        # 4 & 5. Network-Semantic pipelines (require OSM graph)
        # ==============================================================
        print(f"  Downloading OSM graph for {obj_id}...")
        try:
            G = download_osm_graph(raw_points)
            print(f"  Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        except Exception as e:
            print(f"  WARNING: OSM graph download failed: {e}")
            print(f"  Skipping network-semantic pipelines for {obj_id}")
            # Store partial results with NaN for network-semantic metrics
            rec = {
                "obj_id": obj_id,
                "n_raw_points": n_raw,
                **{f"plain_{k}": v for k, v in metrics_plain.items()},
                **{f"oracle_g_{k}": v for k, v in metrics_oracle_g.items()},
                **{f"hysoc_g_{k}": v for k, v in metrics_hysoc_g.items()},
                **{f"oracle_n_{k}": float("nan") for k in metrics_plain.keys()},
                **{f"hysoc_n_{k}": float("nan") for k in metrics_plain.keys()},
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
                        "oracle_n": normalize_pipeline_metrics({}),
                        "hysoc_n": normalize_pipeline_metrics({}),
                    },
                }
            )
            obj_dir = os.path.join(out_dir, obj_id)
            os.makedirs(obj_dir, exist_ok=True)
            with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
                json.dump(rec, f, indent=2)
            continue

        # ==============================================================
        # 4. Oracle-N (STSS + Map Match + STC)
        # ==============================================================
        t0 = time.perf_counter()
        segments_stss_n = stss_oracle.process(raw_points)
        processed_oracle_n = []
        for seg in segments_stss_n:
            if isinstance(seg, Stop):
                processed_oracle_n.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                # Map-match move points, then compress with STC
                matched_move_pts = map_match_points(seg.points, G)
                if matched_move_pts:
                    stc_compressed = stc_oracle.process(Move(points=matched_move_pts))
                    processed_oracle_n.append(Move(points=stc_compressed))
                else:
                    processed_oracle_n.append(Move(points=seg.points))
        t1 = time.perf_counter()
        latency_us_oracle_n = ((t1 - t0) * 1e6) / n_raw
        metrics_oracle_n = compute_segmented_metrics(raw_points, processed_oracle_n, latency_us_oracle_n)
        print(f"  Oracle-N:   CR={metrics_oracle_n['cr']:.2f}  SED={metrics_oracle_n['avg_sed_m']:.2f}m  Lat={latency_us_oracle_n:.1f}µs/pt")

        # ==============================================================
        # 5. HYSOC-N (STEP + Map Match + TRACE) — via HYSOCCompressor
        # ==============================================================
        config_n = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC,
            stop_max_eps_meters=STOP_MAX_EPS_METERS,
            stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
            osm_graph=G,
            enable_map_matching=True,
        )
        compressor_n = HYSOCCompressor(config=config_n)
        t0 = time.perf_counter()
        compressed_n = compressor_n.compress(raw_points)
        t1 = time.perf_counter()
        latency_us_hysoc_n = ((t1 - t0) * 1e6) / n_raw
        metrics_hysoc_n = compute_hysoc_metrics(raw_points, compressed_n, latency_us_hysoc_n)
        print(f"  HYSOC-N:    CR={metrics_hysoc_n['cr']:.2f}  SED={metrics_hysoc_n['avg_sed_m']:.2f}m  Lat={latency_us_hysoc_n:.1f}µs/pt")

        # ------------------------------------------------------------------
        # Collect per-file record
        # ------------------------------------------------------------------
        rec = {
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            **{f"plain_{k}": v for k, v in metrics_plain.items()},
            **{f"oracle_g_{k}": v for k, v in metrics_oracle_g.items()},
            **{f"hysoc_g_{k}": v for k, v in metrics_hysoc_g.items()},
            **{f"oracle_n_{k}": v for k, v in metrics_oracle_n.items()},
            **{f"hysoc_n_{k}": v for k, v in metrics_hysoc_n.items()},
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
                    "oracle_n": normalize_pipeline_metrics(metrics_oracle_n),
                    "hysoc_n": normalize_pipeline_metrics(metrics_hysoc_n),
                },
            }
        )

        # Per-file output
        obj_dir = os.path.join(out_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
            json.dump(rec, f, indent=2)

    # ==================================================================
    # Aggregate CSV
    # ==================================================================
    csv_path = os.path.join(out_dir, "demo21_evaluation_metrics.csv")
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved evaluation aggregate metrics: {csv_path}")

        # Aggregated summary (mean & median), ignoring NaN
        agg_metrics: Dict[str, Dict[str, float]] = {"mean": {}, "median": {}}
        keys_to_agg = [
            "plain_cr", "oracle_g_cr", "hysoc_g_cr", "oracle_n_cr", "hysoc_n_cr",
            "plain_avg_sed_m", "oracle_g_avg_sed_m", "hysoc_g_avg_sed_m",
            "oracle_n_avg_sed_m", "hysoc_n_avg_sed_m",
            "plain_latency_us_per_point", "oracle_g_latency_us_per_point",
            "hysoc_g_latency_us_per_point", "oracle_n_latency_us_per_point",
            "hysoc_n_latency_us_per_point",
        ]
        for key in keys_to_agg:
            vals = [float(r[key]) for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
            if vals:
                agg_metrics["mean"][key] = float(np.mean(vals))
                agg_metrics["median"][key] = float(np.median(vals))
            else:
                agg_metrics["mean"][key] = float("nan")
                agg_metrics["median"][key] = float("nan")

        agg_path = os.path.join(out_dir, "agg_summary.json")
        with open(agg_path, "w", newline="") as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"Saved aggregated statistics: {agg_path}")

    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_21_hysoc_vs_oracles",
        run_config={
            "buffer_capacity": args.buffer_capacity,
            "dp_epsilon_meters": args.dp_epsilon_meters,
            "subset_dir": subset_dir,
            "n_input_files": len(csv_files),
            "n_processed_files": len(results),
        },
        per_file_records=contract_per_file,
        metadata={
            "notes": "Network pipelines may be null when OSM graph download fails.",
        },
    )
    print(f"Saved contract bundle: {contract_paths['contract_agg_summary']}")

    # ==================================================================
    # Summary Boxplots
    # ==================================================================
    if not results:
        print("No results to plot.")
        return

    # Filter out records with NaN network-semantic values for clean plotting
    full_results = [
        r for r in results
        if not (isinstance(r.get("oracle_n_cr"), float) and np.isnan(r["oracle_n_cr"]))
    ]

    if not full_results:
        print("No complete results (with network-semantic) to plot.")
        return

    labels = ["Plain DP", "Oracle-G", "HYSOC-G", "Oracle-N", "HYSOC-N"]

    plain_cr = [r["plain_cr"] for r in full_results]
    oracle_g_cr = [r["oracle_g_cr"] for r in full_results]
    hysoc_g_cr = [r["hysoc_g_cr"] for r in full_results]
    oracle_n_cr = [r["oracle_n_cr"] for r in full_results]
    hysoc_n_cr = [r["hysoc_n_cr"] for r in full_results]

    plain_sed = [r["plain_avg_sed_m"] for r in full_results]
    oracle_g_sed = [r["oracle_g_avg_sed_m"] for r in full_results]
    hysoc_g_sed = [r["hysoc_g_avg_sed_m"] for r in full_results]
    oracle_n_sed = [r["oracle_n_avg_sed_m"] for r in full_results]
    hysoc_n_sed = [r["hysoc_n_avg_sed_m"] for r in full_results]

    plain_lat = [r["plain_latency_us_per_point"] for r in full_results]
    oracle_g_lat = [r["oracle_g_latency_us_per_point"] for r in full_results]
    hysoc_g_lat = [r["hysoc_g_latency_us_per_point"] for r in full_results]
    oracle_n_lat = [r["oracle_n_latency_us_per_point"] for r in full_results]
    hysoc_n_lat = [r["hysoc_n_latency_us_per_point"] for r in full_results]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --- CR ---
    bp1 = axes[0].boxplot(
        [plain_cr, oracle_g_cr, hysoc_g_cr, oracle_n_cr, hysoc_n_cr],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    colors_cr = ["#bdbdbd", "#64b5f6", "#1565c0", "#ffb74d", "#e65100"]
    for patch, color in zip(bp1["boxes"], colors_cr):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Compression Ratio (CR)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Raw / Compressed")
    axes[0].grid(True, alpha=0.3)

    # --- SED ---
    bp2 = axes[1].boxplot(
        [plain_sed, oracle_g_sed, hysoc_g_sed, oracle_n_sed, hysoc_n_sed],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    colors_sed = colors_cr
    for patch, color in zip(bp2["boxes"], colors_sed):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Information Loss (Mean SED)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Synchronized Euclidean Distance (m)")
    axes[1].grid(True, alpha=0.3)

    # --- Latency ---
    bp3 = axes[2].boxplot(
        [plain_lat, oracle_g_lat, hysoc_g_lat, oracle_n_lat, hysoc_n_lat],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    colors_lat = colors_cr
    for patch, color in zip(bp3["boxes"], colors_lat):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_title("Processing Latency (µs/point)", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Latency (µs) — Log Scale")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Demo 21: HYSOC Full Evaluation — {len(full_results)} trajectories",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo21_vs_oracles_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary comparison plot: {plot_path}")

    print("\n" + "=" * 60)
    print("Demo 21 completed.")
    print(f"Results: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
