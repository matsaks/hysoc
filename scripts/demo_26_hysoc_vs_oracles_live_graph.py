# ruff: noqa: E402

"""
Demo 26: HYSOC vs Oracles using live (non-cached) OSM graph download.

Runs the same 5 compression pipelines as Demo 23 on every file in input directory:

  1. Plain DP                     — no segmentation, whole-trajectory DP
  2. Oracle-G (STSS + DP)         — offline geometric oracle
  3. HYSOC-G (STEP + SQUISH/DP)   — online geometric
  4. Oracle-N (STSS + Map + STC)  — offline network-semantic oracle
  5. HYSOC-N (STEP + Map + TRACE) — online network-semantic

Unlike Demo 23, this demo does not load a cached GraphML file.
It downloads one OSM graph from a configurable bounding box at runtime,
then reuses it for all trajectories in the run.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

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
from core.compression import CompressionStrategy, HYSOCConfig
from core.segment import Move, Stop
from engines.dp import DouglasPeuckerCompressor
from engines.squish import SquishCompressor
from engines.stop_compressor import StopCompressor
from oracle.oracleN import OracleN
from oracle.oracleG import OracleG
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

import demo_23_hysoc_vs_oracles_cached_graph as demo23

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_26_hysoc_vs_oracles_live_graph")
DEFAULT_INPUT_DIR = os.path.join("data", "raw", "London_Final_100")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS

# London M25 bbox defaults (same region as demo_22)
DEFAULT_LAT_MIN = 51.28
DEFAULT_LAT_MAX = 51.69
DEFAULT_LON_MIN = -0.51
DEFAULT_LON_MAX = 0.33


def _to_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def _import_osmnx():
    try:
        import osmnx as ox  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "osmnx is required to download OSM graph. Install it with `pip install osmnx`."
        ) from exc
    return ox


def download_graph_from_bbox(
    *, lat_min: float, lat_max: float, lon_min: float, lon_max: float, network_type: str
):
    ox = _import_osmnx()
    t0 = time.perf_counter()
    graph = ox.graph_from_bbox(
        bbox=(lon_min, lat_min, lon_max, lat_max),
        network_type=network_type,
        simplify=True,
    )
    t1 = time.perf_counter()

    meta = {
        "graph_source": "live_osm_download",
        "graph_policy": "download_once_per_run_no_cache_file",
        "graph_download_time_s": float(t1 - t0),
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "bbox": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
        },
        "network_type": network_type,
    }
    return graph, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 26: HYSOC (G+N) vs Oracles using live OSM graph download."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--network-type", type=str, default="drive")
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument("--max-files", type=int, default=0, help="If >0, process only the first N files.")
    parser.add_argument(
        "--no-per-file-plots",
        action="store_true",
        help="Disable per-file plots (enabled by default).",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.lat_min >= args.lat_max:
        raise ValueError("--lat-min must be < --lat-max")
    if args.lon_min >= args.lon_max:
        raise ValueError("--lon-min must be < --lon-max")

    from hysoc.hysocG import HYSOCCompressor

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

    print(f"Running Demo 26 on {len(csv_files)} trajectories in {input_dir}")
    print(
        "Downloading live graph for bbox: "
        f"lat[{args.lat_min}, {args.lat_max}] lon[{args.lon_min}, {args.lon_max}]"
    )
    graph, graph_meta = download_graph_from_bbox(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        network_type=args.network_type,
    )
    print(f"Graph source: {graph_meta['graph_source']}")
    print(f"Graph nodes/edges: {graph_meta['graph_nodes']}/{graph_meta['graph_edges']}")
    print(f"Graph download time: {graph_meta['graph_download_time_s']:.2f} s")
    print(f"Output directory: {out_dir}\n")

    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    stss_oracle = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    stc_oracle = OracleN()

    results: List[Dict[str, Any]] = []
    contract_per_file: List[Dict[str, Any]] = []
    online_processing_time_s = 0.0

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(input_dir, fname)
        raw_points = demo23.load_trajectory(path, obj_id=obj_id)
        if len(raw_points) < 2:
            print(f"[{file_idx}/{len(csv_files)}] {obj_id}: skipped (<2 points)")
            continue

        n_raw = len(raw_points)
        print(f"\n[{file_idx}/{len(csv_files)}] {obj_id}: {n_raw} points")

        t0 = time.perf_counter()
        plain_dp_points = dp_compressor.compress(raw_points)
        processed_plain = [Move(points=plain_dp_points)]
        t1 = time.perf_counter()
        online_processing_time_s += float(t1 - t0)
        latency_us_plain = ((t1 - t0) * 1e6) / n_raw
        metrics_plain = demo23.compute_segmented_metrics(raw_points, processed_plain, latency_us_plain)

        t0 = time.perf_counter()
        segments_stss = stss_oracle.process(raw_points)
        processed_oracle_g = []
        for seg in segments_stss:
            if isinstance(seg, Stop):
                processed_oracle_g.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                processed_oracle_g.append(Move(points=dp_compressor.compress(seg.points)))
        t1 = time.perf_counter()
        online_processing_time_s += float(t1 - t0)
        latency_us_oracle_g = ((t1 - t0) * 1e6) / n_raw
        metrics_oracle_g = demo23.compute_segmented_metrics(
            raw_points, processed_oracle_g, latency_us_oracle_g
        )

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
        online_processing_time_s += float(t1 - t0)
        latency_us_hysoc_g = ((t1 - t0) * 1e6) / n_raw
        metrics_hysoc_g = demo23.compute_hysoc_metrics(raw_points, compressed_g, latency_us_hysoc_g)

        t0 = time.perf_counter()
        segments_stss_n = stss_oracle.process(raw_points)
        processed_oracle_n = []
        for seg in segments_stss_n:
            if isinstance(seg, Stop):
                processed_oracle_n.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                matched_move_pts = demo23.map_match_points(seg.points, graph)
                if matched_move_pts:
                    stc_compressed = stc_oracle.process(Move(points=matched_move_pts))
                    processed_oracle_n.append(Move(points=stc_compressed))
                else:
                    processed_oracle_n.append(Move(points=seg.points))
        t1 = time.perf_counter()
        online_processing_time_s += float(t1 - t0)
        latency_us_oracle_n = ((t1 - t0) * 1e6) / n_raw
        metrics_oracle_n = demo23.compute_segmented_metrics(
            raw_points, processed_oracle_n, latency_us_oracle_n
        )

        config_n = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC,
            stop_max_eps_meters=STOP_MAX_EPS_METERS,
            stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
            osm_graph=graph,
            enable_map_matching=True,
        )
        compressor_n = HYSOCCompressor(config=config_n)
        t0 = time.perf_counter()
        compressed_n = compressor_n.compress(raw_points)
        t1 = time.perf_counter()
        online_processing_time_s += float(t1 - t0)
        latency_us_hysoc_n = ((t1 - t0) * 1e6) / n_raw
        metrics_hysoc_n = demo23.compute_hysoc_metrics(raw_points, compressed_n, latency_us_hysoc_n)

        print(
            "  "
            f"Plain={metrics_plain['cr']:.2f}/{metrics_plain['avg_sed_m']:.2f}m, "
            f"Oracle-G={metrics_oracle_g['cr']:.2f}/{metrics_oracle_g['avg_sed_m']:.2f}m, "
            f"HYSOC-G={metrics_hysoc_g['cr']:.2f}/{metrics_hysoc_g['avg_sed_m']:.2f}m, "
            f"Oracle-N={metrics_oracle_n['cr']:.2f}/{metrics_oracle_n['avg_sed_m']:.2f}m, "
            f"HYSOC-N={metrics_hysoc_n['cr']:.2f}/{metrics_hysoc_n['avg_sed_m']:.2f}m"
        )

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

        obj_dir = os.path.join(out_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)
        with open(os.path.join(obj_dir, "metrics.json"), "w", newline="") as f:
            json.dump(rec, f, indent=2)

        if not args.no_per_file_plots:
            try:
                hysoc_g_items = demo23.compressed_trajectory_to_items(compressed_g)
                hysoc_n_items = demo23.compressed_trajectory_to_items(compressed_n)
                demo23.plot_trajectory_comparison(
                    out_path=os.path.join(obj_dir, "trajectory_comparison.png"),
                    raw_points=raw_points,
                    plain_dp_items=processed_plain,
                    oracle_g_items=processed_oracle_g,
                    hysoc_g_items=hysoc_g_items,
                    oracle_n_items=processed_oracle_n,
                    hysoc_n_items=hysoc_n_items,
                    title=f"Trajectory {obj_id} (Demo 26 Full Pipeline Comparison)",
                )
                demo23.plot_per_file_pipeline(
                    out_path=os.path.join(obj_dir, "pipeline_comparison.png"),
                    obj_id=obj_id,
                    metrics_plain=metrics_plain,
                    metrics_oracle_g=metrics_oracle_g,
                    metrics_hysoc_g=metrics_hysoc_g,
                    metrics_oracle_n=metrics_oracle_n,
                    metrics_hysoc_n=metrics_hysoc_n,
                )
            except Exception as e:
                print(f"  Plot skipped for {obj_id}: {e}")

    if results:
        csv_path = os.path.join(out_dir, "demo26_evaluation_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved aggregate metrics CSV: {csv_path}")

        agg_metrics: Dict[str, Any] = {"mean": {}, "median": {}, "graph": graph_meta}
        keys_to_agg = [
            "plain_cr",
            "oracle_g_cr",
            "hysoc_g_cr",
            "oracle_n_cr",
            "hysoc_n_cr",
            "plain_avg_sed_m",
            "oracle_g_avg_sed_m",
            "hysoc_g_avg_sed_m",
            "oracle_n_avg_sed_m",
            "hysoc_n_avg_sed_m",
            "plain_latency_us_per_point",
            "oracle_g_latency_us_per_point",
            "hysoc_g_latency_us_per_point",
            "oracle_n_latency_us_per_point",
            "hysoc_n_latency_us_per_point",
        ]
        for key in keys_to_agg:
            vals = [float(r[key]) for r in results]
            agg_metrics["mean"][key] = float(np.mean(vals)) if vals else float("nan")
            agg_metrics["median"][key] = float(np.median(vals)) if vals else float("nan")

        provisioning_time_s = float(graph_meta["graph_download_time_s"])
        end_to_end_time_s = provisioning_time_s + online_processing_time_s
        agg_metrics["timing"] = {
            "provisioning_time_s": provisioning_time_s,
            "online_processing_time_s": online_processing_time_s,
            "end_to_end_time_s": end_to_end_time_s,
            "latency_policy": "online_primary_with_end_to_end_secondary",
        }

        agg_path = os.path.join(out_dir, "agg_summary.json")
        with open(agg_path, "w", newline="") as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"Saved aggregated summary: {agg_path}")

    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_26_hysoc_vs_oracles_live_graph",
        run_config={
            "input_dir": input_dir,
            "graph_source": graph_meta["graph_source"],
            "graph_nodes": graph_meta["graph_nodes"],
            "graph_edges": graph_meta["graph_edges"],
            "graph_download_time_s": graph_meta["graph_download_time_s"],
            "graph_policy": graph_meta["graph_policy"],
            "bbox": graph_meta["bbox"],
            "network_type": graph_meta["network_type"],
            "buffer_capacity": args.buffer_capacity,
            "dp_epsilon_meters": args.dp_epsilon_meters,
            "n_input_files": len(csv_files),
            "n_processed_files": len(results),
            "provisioning_time_s": graph_meta["graph_download_time_s"],
            "online_processing_time_s": online_processing_time_s,
            "end_to_end_time_s": graph_meta["graph_download_time_s"] + online_processing_time_s,
            "latency_policy": "online_primary_with_end_to_end_secondary",
        },
        per_file_records=contract_per_file,
        metadata={
            "notes": "Live graph download from OSM bbox at runtime; no GraphML cache file is used.",
        },
    )
    print(f"Saved contract bundle: {contract_paths['contract_agg_summary']}")

    if not results:
        print("No results to plot.")
        return

    labels = ["Plain DP", "Oracle-G", "HYSOC-G", "Oracle-N", "HYSOC-N"]
    plain_cr = [r["plain_cr"] for r in results]
    oracle_g_cr = [r["oracle_g_cr"] for r in results]
    hysoc_g_cr = [r["hysoc_g_cr"] for r in results]
    oracle_n_cr = [r["oracle_n_cr"] for r in results]
    hysoc_n_cr = [r["hysoc_n_cr"] for r in results]

    plain_sed = [r["plain_avg_sed_m"] for r in results]
    oracle_g_sed = [r["oracle_g_avg_sed_m"] for r in results]
    hysoc_g_sed = [r["hysoc_g_avg_sed_m"] for r in results]
    oracle_n_sed = [r["oracle_n_avg_sed_m"] for r in results]
    hysoc_n_sed = [r["hysoc_n_avg_sed_m"] for r in results]

    plain_lat = [r["plain_latency_us_per_point"] for r in results]
    oracle_g_lat = [r["oracle_g_latency_us_per_point"] for r in results]
    hysoc_g_lat = [r["hysoc_g_latency_us_per_point"] for r in results]
    oracle_n_lat = [r["oracle_n_latency_us_per_point"] for r in results]
    hysoc_n_lat = [r["hysoc_n_latency_us_per_point"] for r in results]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    colors = ["#bdbdbd", "#64b5f6", "#1565c0", "#ffb74d", "#e65100"]

    bp1 = axes[0].boxplot(
        [plain_cr, oracle_g_cr, hysoc_g_cr, oracle_n_cr, hysoc_n_cr],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_title("Compression Ratio (CR)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Raw / Compressed")
    axes[0].grid(True, alpha=0.3)

    bp2 = axes[1].boxplot(
        [plain_sed, oracle_g_sed, hysoc_g_sed, oracle_n_sed, hysoc_n_sed],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_title("Information Loss (Mean SED)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("SED (m)")
    axes[1].grid(True, alpha=0.3)

    bp3 = axes[2].boxplot(
        [plain_lat, oracle_g_lat, hysoc_g_lat, oracle_n_lat, hysoc_n_lat],
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )
    for patch, color in zip(bp3["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[2].set_title("Processing Latency (us/point)", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("us (log scale)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Demo 26: HYSOC Full Evaluation (live graph) — {len(results)} trajectories",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo26_vs_oracles_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary comparison plot: {plot_path}")

    print("\n" + "=" * 60)
    print("Demo 26 completed.")
    print(f"Results: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

