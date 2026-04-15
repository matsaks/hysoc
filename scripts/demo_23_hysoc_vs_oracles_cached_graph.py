# ruff: noqa: E402

"""
Demo 23: HYSOC vs Oracles using cached London M25 graph.

Runs 5 compression pipelines on every file in input directory:

  1. Plain DP                     — no segmentation, whole-trajectory DP
  2. Oracle-G (STSS + DP)         — offline geometric oracle
  3. HYSOC-G (STEP + SQUISH/DP)   — online geometric
  4. Oracle-N (STSS + Map + STC)  — offline network-semantic oracle
  5. HYSOC-N (STEP + Map + TRACE) — online network-semantic

Compared to demo_21, this demo never downloads OSM data.
It loads one pre-cached graph from GraphML and reuses it for all trajectories.
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

from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from hysoc.constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from hysoc.core.compression import CompressionStrategy, HYSOCConfig
from hysoc.core.point import Point
from hysoc.core.segment import Move, Stop
from hysoc.metrics import calculate_sed_stats
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor
from hysoc.modules.move_compression.squish import SquishCompressor
from hysoc.modules.stop_compression.compressor import CompressedStop, StopCompressor
from benchmarks.oracles.stc import STCOracle
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_23_hysoc_vs_oracles_cached_graph")
DEFAULT_INPUT_DIR = os.path.join("data", "raw", "London_Final_100")
DEFAULT_GRAPH_PATH = os.path.join("data", "processed", "osm_graphs", "london_m25_drive.graphml")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS


def _import_osmnx():
    try:
        import osmnx as ox  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "osmnx is required to load cached GraphML. Install it with `pip install osmnx`."
        ) from exc
    return ox


def _to_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def load_cached_graph(graph_path: str) -> Tuple[Any, Dict[str, Any]]:
    graph_abs = _to_abs_path(graph_path)
    if not os.path.exists(graph_abs):
        raise FileNotFoundError(
            f"Cached graph not found: {graph_abs}\n"
            "Prepare it first with scripts/demo_22_prepare_london_m25_graph.py"
        )

    ox = _import_osmnx()
    t0 = time.perf_counter()
    graph = ox.load_graphml(graph_abs)
    t1 = time.perf_counter()

    size_bytes = os.path.getsize(graph_abs)
    meta = {
        "graph_path": graph_abs,
        "graph_file_size_bytes": int(size_bytes),
        "graph_file_size_mb": float(size_bytes / (1024 * 1024)),
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "graph_load_time_s": float(t1 - t0),
        "graph_source": "cached_graphml",
        "graph_policy": "cached_graph_only_no_network_download",
    }
    return graph, meta


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
        "latency_us_per_point": latency_us,
    }


def compute_hysoc_metrics(original: List[Point], compressed_trajectory, latency_us: float) -> Dict[str, Any]:
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


def map_match_points(raw_points: List[Point], graph) -> List[Point]:
    from hysoc.modules.map_matching.matcher import OnlineMapMatcher

    matcher = OnlineMapMatcher(graph)
    matched: List[Point] = []
    for p in raw_points:
        result = matcher.process_point(p)
        if result is not None:
            matched.append(result)
    matched.extend(matcher.flush())
    return matched


def compressed_trajectory_to_items(compressed_trajectory) -> List[object]:
    """Convert CompressedTrajectory segments into Stop/Move-like items for plotting."""
    items: List[object] = []
    for seg in compressed_trajectory.compressed_segments:
        data = seg.compressed_data
        if seg.segment_type == "stop":
            items.append(data)
            continue

        move_points: List[Point] = []
        if isinstance(data, list):
            move_points = data
        elif isinstance(data, dict) and "retained_points" in data:
            move_points = data["retained_points"]
        elif hasattr(data, "points") and data.points:
            move_points = data.points

        items.append(Move(points=move_points))
    return items


def plot_trajectory_comparison(
    out_path: str,
    raw_points: List[Point],
    plain_dp_items: List[object],
    oracle_g_items: List[object],
    hysoc_g_items: List[object],
    oracle_n_items: List[object],
    hysoc_n_items: List[object],
    title: str,
) -> None:
    """Map-style trajectory comparison including both G and N pipelines."""
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point as ShapelyPoint

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
    oracle_g_move_pts, oracle_g_stops = items_to_geom(oracle_g_items)
    hysoc_g_move_pts, hysoc_g_stops = items_to_geom(hysoc_g_items)
    oracle_n_move_pts, oracle_n_stops = items_to_geom(oracle_n_items)
    hysoc_n_move_pts, hysoc_n_stops = items_to_geom(hysoc_n_items)

    fig, axes = plt.subplots(1, 6, figsize=(34, 6), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.02)
    ax_raw, ax_plain, ax_oracle_g, ax_hysoc_g, ax_oracle_n, ax_hysoc_n = axes.ravel()

    def plot_panel(ax, moves, stops, comp_color, is_raw=False):
        if raw_pts:
            gpd.GeoSeries(raw_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color="gray", markersize=5, alpha=0.35
            )
        if is_raw and raw_pts:
            gpd.GeoSeries(raw_pts, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color=comp_color, markersize=12, alpha=0.9
            )
        if moves and not is_raw:
            gpd.GeoSeries(moves, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color=comp_color, markersize=12, alpha=0.9, label="Move Point"
            )
        if stops and not is_raw:
            gpd.GeoSeries(stops, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color="black", markersize=30, marker="o", zorder=5, label="Stop Centroid"
            )
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
    plot_panel(ax_oracle_g, oracle_g_move_pts, oracle_g_stops, "blue")
    ax_oracle_g.set_title("Oracle-G (STSS + DP)", fontsize=14)
    plot_panel(ax_hysoc_g, hysoc_g_move_pts, hysoc_g_stops, "green")
    ax_hysoc_g.set_title("HYSOC-G (STEP + SQUISH/DP)", fontsize=14)
    plot_panel(ax_oracle_n, oracle_n_move_pts, oracle_n_stops, "#fb8c00")
    ax_oracle_n.set_title("Oracle-N (STSS + STC)", fontsize=14)
    plot_panel(ax_hysoc_n, hysoc_n_move_pts, hysoc_n_stops, "#6a1b9a")
    ax_hysoc_n.set_title("HYSOC-N (STEP + TRACE)", fontsize=14)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_file_pipeline(
    out_path: str,
    obj_id: str,
    metrics_plain: Dict[str, Any],
    metrics_oracle_g: Dict[str, Any],
    metrics_hysoc_g: Dict[str, Any],
    metrics_oracle_n: Dict[str, Any],
    metrics_hysoc_n: Dict[str, Any],
) -> None:
    labels = ["Plain DP", "Oracle-G", "HYSOC-G", "Oracle-N", "HYSOC-N"]
    cr = [
        metrics_plain["cr"],
        metrics_oracle_g["cr"],
        metrics_hysoc_g["cr"],
        metrics_oracle_n["cr"],
        metrics_hysoc_n["cr"],
    ]
    sed = [
        metrics_plain["avg_sed_m"],
        metrics_oracle_g["avg_sed_m"],
        metrics_hysoc_g["avg_sed_m"],
        metrics_oracle_n["avg_sed_m"],
        metrics_hysoc_n["avg_sed_m"],
    ]
    lat = [
        metrics_plain["latency_us_per_point"],
        metrics_oracle_g["latency_us_per_point"],
        metrics_hysoc_g["latency_us_per_point"],
        metrics_oracle_n["latency_us_per_point"],
        metrics_hysoc_n["latency_us_per_point"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#bdbdbd", "#64b5f6", "#1565c0", "#ffb74d", "#e65100"]

    axes[0].bar(labels, cr, color=colors, alpha=0.85)
    axes[0].set_title("CR")
    axes[0].set_ylabel("Raw / Compressed")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(labels, sed, color=colors, alpha=0.85)
    axes[1].set_title("Mean SED (m)")
    axes[1].set_ylabel("m")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(labels, lat, color=colors, alpha=0.85)
    axes[2].set_title("Latency (us/point)")
    axes[2].set_ylabel("us")
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"Demo 23 per-file comparison: {obj_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 23: HYSOC (G+N) vs Oracles using cached London graph."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--graph-path", default=DEFAULT_GRAPH_PATH)
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

    from hysoc.modules.hysoc import HYSOCCompressor

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

    graph, graph_meta = load_cached_graph(args.graph_path)

    print(f"Running Demo 23 on {len(csv_files)} trajectories in {input_dir}")
    print(f"Using cached graph: {graph_meta['graph_path']}")
    print(
        f"Graph size: {graph_meta['graph_file_size_mb']:.2f} MB "
        f"({graph_meta['graph_file_size_bytes']} bytes)"
    )
    print(f"Graph nodes/edges: {graph_meta['graph_nodes']}/{graph_meta['graph_edges']}")
    print(f"Graph load time: {graph_meta['graph_load_time_s']:.2f} s")
    print(f"Output directory: {out_dir}\n")

    stop_compressor = StopCompressor()
    squish = SquishCompressor(capacity=args.buffer_capacity)
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    stss_oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    stc_oracle = STCOracle()

    results: List[Dict[str, Any]] = []
    contract_per_file: List[Dict[str, Any]] = []
    online_processing_time_s = 0.0

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path = os.path.join(input_dir, fname)
        raw_points = load_trajectory(path, obj_id=obj_id)
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
        metrics_plain = compute_segmented_metrics(raw_points, processed_plain, latency_us_plain)

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
        metrics_oracle_g = compute_segmented_metrics(raw_points, processed_oracle_g, latency_us_oracle_g)

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
        metrics_hysoc_g = compute_hysoc_metrics(raw_points, compressed_g, latency_us_hysoc_g)

        t0 = time.perf_counter()
        segments_stss_n = stss_oracle.process(raw_points)
        processed_oracle_n = []
        for seg in segments_stss_n:
            if isinstance(seg, Stop):
                processed_oracle_n.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                matched_move_pts = map_match_points(seg.points, graph)
                if matched_move_pts:
                    stc_compressed = stc_oracle.process(Move(points=matched_move_pts))
                    processed_oracle_n.append(Move(points=stc_compressed))
                else:
                    processed_oracle_n.append(Move(points=seg.points))
        t1 = time.perf_counter()
        online_processing_time_s += float(t1 - t0)
        latency_us_oracle_n = ((t1 - t0) * 1e6) / n_raw
        metrics_oracle_n = compute_segmented_metrics(raw_points, processed_oracle_n, latency_us_oracle_n)

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
        metrics_hysoc_n = compute_hysoc_metrics(raw_points, compressed_n, latency_us_hysoc_n)

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
                hysoc_g_items = compressed_trajectory_to_items(compressed_g)
                # Keep original stop/move segment structure so stop centroids are visible.
                hysoc_n_items = compressed_trajectory_to_items(compressed_n)
                plot_trajectory_comparison(
                    out_path=os.path.join(obj_dir, "trajectory_comparison.png"),
                    raw_points=raw_points,
                    plain_dp_items=processed_plain,
                    oracle_g_items=processed_oracle_g,
                    hysoc_g_items=hysoc_g_items,
                    oracle_n_items=processed_oracle_n,
                    hysoc_n_items=hysoc_n_items,
                    title=f"Trajectory {obj_id} (Demo 23 Full Pipeline Comparison)",
                )
                plot_per_file_pipeline(
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
        csv_path = os.path.join(out_dir, "demo23_evaluation_metrics.csv")
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

        provisioning_time_s = float(graph_meta["graph_load_time_s"])
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
        script_name="demo_23_hysoc_vs_oracles_cached_graph",
        run_config={
            "input_dir": input_dir,
            "graph_path": graph_meta["graph_path"],
            "graph_file_size_bytes": graph_meta["graph_file_size_bytes"],
            "graph_file_size_mb": graph_meta["graph_file_size_mb"],
            "graph_nodes": graph_meta["graph_nodes"],
            "graph_edges": graph_meta["graph_edges"],
            "graph_load_time_s": graph_meta["graph_load_time_s"],
            "graph_policy": graph_meta["graph_policy"],
            "buffer_capacity": args.buffer_capacity,
            "dp_epsilon_meters": args.dp_epsilon_meters,
            "n_input_files": len(csv_files),
            "n_processed_files": len(results),
            "provisioning_time_s": graph_meta["graph_load_time_s"],
            "online_processing_time_s": online_processing_time_s,
            "end_to_end_time_s": graph_meta["graph_load_time_s"] + online_processing_time_s,
            "latency_policy": "online_primary_with_end_to_end_secondary",
        },
        per_file_records=contract_per_file,
        metadata={
            "notes": "No network OSM download; cached graph is reused for all files.",
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
        f"Demo 23: HYSOC Full Evaluation (cached graph) — {len(results)} trajectories",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "demo23_vs_oracles_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary comparison plot: {plot_path}")

    print("\n" + "=" * 60)
    print("Demo 23 completed.")
    print(f"Results: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

