# ruff: noqa: E402
"""
Demo 25: Oracle-N and HYSOC-N diagnostics on one London trajectory.

Investigates:
  - Where latency is spent (STSS, map matching, STC, HYSOC internals).
  - Why Mean SED is high (aggressive retention and long interpolation spans).
"""

import argparse
import csv
import json
import math
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

from benchmarks.oracles.stc import STCOracle
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from hysoc.constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from hysoc.core.compression import CompressionStrategy, HYSOCConfig
from hysoc.core.point import Point
from hysoc.core.segment import Move, Stop
from hysoc.metrics.sed import calculate_sed_error, calculate_sed_stats
from hysoc.modules.stop_compression.compressor import CompressedStop, StopCompressor

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_25_oracle_n_hysoc_n_diagnostics")
DEFAULT_INPUT_DIR = os.path.join("data", "raw", "London_Final_100")
DEFAULT_GRAPH_PATH = os.path.join("data", "processed", "osm_graphs", "london_m25_drive.graphml")


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
    return graph, {
        "graph_path": graph_abs,
        "graph_file_size_bytes": int(size_bytes),
        "graph_file_size_mb": float(size_bytes / (1024 * 1024)),
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "graph_load_time_s": float(t1 - t0),
    }


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


def map_match_points_with_diag(raw_points: List[Point], graph) -> Tuple[List[Point], Dict[str, Any]]:
    from hysoc.modules.map_matching.matcher import OnlineMapMatcher

    matcher = OnlineMapMatcher(graph)
    matched: List[Point] = []
    for p in raw_points:
        result = matcher.process_point(p)
        if result is not None:
            matched.append(result)
    matched.extend(matcher.flush())
    diag = matcher.get_diagnostics() if hasattr(matcher, "get_diagnostics") else {}
    return matched, diag


def compressed_trajectory_to_items(compressed_trajectory) -> List[object]:
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


def calculate_sed_outliers(
    original: List[Point],
    compressed: List[Point],
    top_k: int,
) -> Dict[str, Any]:
    if not original or not compressed:
        return {"top_k": [], "segment_span_stats": {"mean_segment_dt_s": 0.0, "max_segment_dt_s": 0.0}}

    outliers: List[Dict[str, Any]] = []
    seg_spans: List[float] = []
    comp_idx = 0
    for i, p in enumerate(original):
        while comp_idx < len(compressed) - 1 and p.timestamp > compressed[comp_idx + 1].timestamp:
            comp_idx += 1
        if comp_idx >= len(compressed) - 1:
            continue
        p_start = compressed[comp_idx]
        p_end = compressed[comp_idx + 1]
        err = calculate_sed_error(p, p_start, p_end)
        seg_dt = (p_end.timestamp - p_start.timestamp).total_seconds()
        seg_spans.append(seg_dt)
        outliers.append(
            {
                "original_index": i,
                "original_time": p.timestamp.isoformat(),
                "original_lat": p.lat,
                "original_lon": p.lon,
                "sed_m": float(err),
                "segment_start_time": p_start.timestamp.isoformat(),
                "segment_end_time": p_end.timestamp.isoformat(),
                "segment_dt_s": float(seg_dt),
            }
        )
    outliers.sort(key=lambda r: r["sed_m"], reverse=True)
    top = outliers[:top_k]
    if seg_spans:
        span_stats = {
            "mean_segment_dt_s": float(np.mean(seg_spans)),
            "p95_segment_dt_s": float(np.percentile(np.asarray(seg_spans, dtype=float), 95)),
            "max_segment_dt_s": float(np.max(seg_spans)),
        }
    else:
        span_stats = {"mean_segment_dt_s": 0.0, "p95_segment_dt_s": 0.0, "max_segment_dt_s": 0.0}
    return {"top_k": top, "segment_span_stats": span_stats}


def compute_segmented_metrics_with_diag(
    original: List[Point], items: List[object], latency_us: float, top_k: int
) -> Dict[str, Any]:
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
    sed_diag = calculate_sed_outliers(original, sed_stream, top_k=top_k)
    return {
        "cr": float(cr),
        "stored_points": int(stored_points),
        "avg_sed_m": float(avg_sed),
        "p95_sed_m": float(p95_sed),
        "max_sed_m": float(max_sed),
        "latency_us_per_point": float(latency_us),
        "sed_diagnostics": sed_diag,
    }


def compute_hysoc_metrics_with_diag(original: List[Point], compressed_trajectory, latency_us: float, top_k: int) -> Dict[str, Any]:
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
    sed_diag = calculate_sed_outliers(original, reconstructed, top_k=top_k)
    return {
        "cr": float(cr),
        "stored_points": int(stored_points),
        "avg_sed_m": float(avg_sed),
        "p95_sed_m": float(p95_sed),
        "max_sed_m": float(max_sed),
        "latency_us_per_point": float(latency_us),
        "sed_diagnostics": sed_diag,
    }


def _distance_m(p1: Point, p2: Point) -> float:
    d_lat = p1.lat - p2.lat
    d_lon = p1.lon - p2.lon
    avg_lat = math.radians((p1.lat + p2.lat) / 2.0)
    d_lat_m = d_lat * 111320.0
    d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
    return math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m)


def retention_diagnostics_from_moves(items: List[object], raw_points: List[Point]) -> Dict[str, Any]:
    retained_points: List[Point] = []
    for item in items:
        if isinstance(item, Move):
            retained_points.extend(item.points)
    retained_points = sorted(retained_points, key=lambda p: p.timestamp)
    if len(retained_points) < 2:
        return {
            "retained_points": len(retained_points),
            "retention_ratio": float(len(retained_points) / max(1, len(raw_points))),
            "mean_retained_dt_s": 0.0,
            "max_retained_dt_s": 0.0,
            "mean_retained_jump_m": 0.0,
            "max_retained_jump_m": 0.0,
        }
    dts = [
        (retained_points[i + 1].timestamp - retained_points[i].timestamp).total_seconds()
        for i in range(len(retained_points) - 1)
    ]
    dists = [_distance_m(retained_points[i], retained_points[i + 1]) for i in range(len(retained_points) - 1)]
    return {
        "retained_points": len(retained_points),
        "retention_ratio": float(len(retained_points) / max(1, len(raw_points))),
        "mean_retained_dt_s": float(np.mean(dts)),
        "p95_retained_dt_s": float(np.percentile(np.asarray(dts, dtype=float), 95)),
        "max_retained_dt_s": float(np.max(dts)),
        "mean_retained_jump_m": float(np.mean(dists)),
        "p95_retained_jump_m": float(np.percentile(np.asarray(dists, dtype=float), 95)),
        "max_retained_jump_m": float(np.max(dists)),
    }


def plot_pipeline_breakdown(out_path: str, oracle_stages_s: Dict[str, float], hysoc_diag: Dict[str, Any]) -> None:
    labels = ["Segmentation", "MapMatching", "Compression"]
    oracle_vals = [
        float(oracle_stages_s.get("stss_s", 0.0)),
        float(oracle_stages_s.get("map_matching_s", 0.0)),
        float(oracle_stages_s.get("stc_s", 0.0)),
    ]
    hysoc_vals = [
        float(hysoc_diag.get("segmentation_time_s", 0.0)),
        float(hysoc_diag.get("map_matching_time_s", 0.0)),
        float(hysoc_diag.get("compression_time_s", 0.0)),
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, oracle_vals, width=width, label="Oracle-N")
    ax.bar(x + width / 2, hysoc_vals, width=width, label="HYSOC-N")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Demo 25 stage timing breakdown")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_retention_vs_sed(
    out_path: str,
    oracle_metrics: Dict[str, Any],
    hysoc_metrics: Dict[str, Any],
) -> None:
    labels = ["Oracle-N", "HYSOC-N"]
    retention = [float(oracle_metrics["retention"]["retention_ratio"]), float(hysoc_metrics["retention"]["retention_ratio"])]
    mean_sed = [float(oracle_metrics["metrics"]["avg_sed_m"]), float(hysoc_metrics["metrics"]["avg_sed_m"])]
    p95_seg_dt = [
        float(oracle_metrics["metrics"]["sed_diagnostics"]["segment_span_stats"].get("p95_segment_dt_s", 0.0)),
        float(hysoc_metrics["metrics"]["sed_diagnostics"]["segment_span_stats"].get("p95_segment_dt_s", 0.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(labels, retention, color=["#fb8c00", "#6a1b9a"])
    axes[0].set_title("Retention ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(labels, mean_sed, color=["#fb8c00", "#6a1b9a"])
    axes[1].set_title("Mean SED (m)")
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(labels, p95_seg_dt, color=["#fb8c00", "#6a1b9a"])
    axes[2].set_title("P95 compressed span (s)")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_overlay(out_path: str, raw_points: List[Point], oracle_items: List[object], hysoc_items: List[object]) -> None:
    raw_x = [p.lon for p in raw_points]
    raw_y = [p.lat for p in raw_points]

    def retained_xy(items: List[object]) -> Tuple[List[float], List[float]]:
        pts: List[Point] = []
        for item in items:
            if isinstance(item, Move):
                pts.extend(item.points)
        pts = sorted(pts, key=lambda p: p.timestamp)
        return [p.lon for p in pts], [p.lat for p in pts]

    ox, oy = retained_xy(oracle_items)
    hx, hy = retained_xy(hysoc_items)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(raw_x, raw_y, color="gray", linewidth=1.0, alpha=0.5, label=f"Raw ({len(raw_points)})")
    ax.scatter(ox, oy, s=16, alpha=0.9, color="#fb8c00", label=f"Oracle-N retained ({len(ox)})")
    ax.scatter(hx, hy, s=14, alpha=0.9, color="#6a1b9a", label=f"HYSOC-N retained ({len(hx)})")
    ax.set_title("Retained points overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 25: Oracle-N and HYSOC-N diagnostics on one file.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--graph-path", default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--file-id", required=True, help="Numeric trajectory id, e.g. 4611343.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-overlay-plot", action="store_true")
    args = parser.parse_args()

    from hysoc.modules.hysoc import HYSOCCompressor

    input_dir = _to_abs_path(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    file_name = f"{args.file_id}.csv"
    file_path = os.path.join(input_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input trajectory not found: {file_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_root, timestamp, str(args.file_id))
    os.makedirs(out_dir, exist_ok=True)

    graph, graph_meta = load_cached_graph(args.graph_path)
    raw_points = load_trajectory(file_path, obj_id=str(args.file_id))
    if len(raw_points) < 2:
        raise ValueError("Trajectory must contain at least 2 points.")
    n_raw = len(raw_points)

    print(f"Demo 25 | file={args.file_id} | points={n_raw}")
    print(f"Graph: {graph_meta['graph_path']}")
    print(f"Output: {out_dir}")

    stop_compressor = StopCompressor()
    stss_oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    stc_oracle = STCOracle()

    # Oracle-N with stage timings and map-matcher diagnostics.
    oracle_stages_s = {"stss_s": 0.0, "map_matching_s": 0.0, "stc_s": 0.0}
    t_oracle_0 = time.perf_counter()
    t0 = time.perf_counter()
    segments_stss = stss_oracle.process(raw_points)
    t1 = time.perf_counter()
    oracle_stages_s["stss_s"] = float(t1 - t0)
    processed_oracle_n: List[object] = []
    oracle_map_diag_acc: Dict[str, float] = {
        "points_in": 0.0,
        "window_wait_count": 0.0,
        "match_window_calls": 0.0,
        "matcher_build_time_s": 0.0,
        "viterbi_match_time_s": 0.0,
        "edge_snap_time_s": 0.0,
        "flush_matches": 0.0,
        "failed_matches": 0.0,
    }
    for seg in segments_stss:
        if isinstance(seg, Stop):
            processed_oracle_n.append(stop_compressor.compress(seg.points))
            continue
        if not isinstance(seg, Move):
            continue
        t0 = time.perf_counter()
        matched_move_pts, mm_diag = map_match_points_with_diag(seg.points, graph)
        t1 = time.perf_counter()
        oracle_stages_s["map_matching_s"] += float(t1 - t0)
        for key in oracle_map_diag_acc:
            oracle_map_diag_acc[key] += float(mm_diag.get(key, 0.0))

        if matched_move_pts:
            t0 = time.perf_counter()
            stc_compressed = stc_oracle.process(Move(points=matched_move_pts))
            t1 = time.perf_counter()
            oracle_stages_s["stc_s"] += float(t1 - t0)
            processed_oracle_n.append(Move(points=stc_compressed))
        else:
            processed_oracle_n.append(Move(points=seg.points))
    t_oracle_1 = time.perf_counter()
    latency_us_oracle_n = ((t_oracle_1 - t_oracle_0) * 1e6) / n_raw
    metrics_oracle_n = compute_segmented_metrics_with_diag(
        raw_points, processed_oracle_n, latency_us_oracle_n, top_k=args.top_k
    )
    oracle_retention = retention_diagnostics_from_moves(processed_oracle_n, raw_points)

    # HYSOC-N with internal diagnostics.
    config_n = HYSOCConfig(
        move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC,
        stop_max_eps_meters=STOP_MAX_EPS_METERS,
        stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        osm_graph=graph,
        enable_map_matching=True,
    )
    compressor_n = HYSOCCompressor(config=config_n)
    t_hysoc_0 = time.perf_counter()
    compressed_n = compressor_n.compress(raw_points)
    t_hysoc_1 = time.perf_counter()
    latency_us_hysoc_n = ((t_hysoc_1 - t_hysoc_0) * 1e6) / n_raw
    metrics_hysoc_n = compute_hysoc_metrics_with_diag(raw_points, compressed_n, latency_us_hysoc_n, top_k=args.top_k)
    hysoc_items = compressed_trajectory_to_items(compressed_n)
    hysoc_retention = retention_diagnostics_from_moves(hysoc_items, raw_points)
    hysoc_diag = compressor_n.get_diagnostics() if hasattr(compressor_n, "get_diagnostics") else {}

    diagnostics = {
        "demo": "demo_25_oracle_n_hysoc_n_diagnostics",
        "obj_id": str(args.file_id),
        "n_raw_points": int(n_raw),
        "graph": graph_meta,
        "oracle_n": {
            "metrics": metrics_oracle_n,
            "stage_timings_s": oracle_stages_s,
            "retention": oracle_retention,
            "map_matcher_diagnostics": oracle_map_diag_acc,
        },
        "hysoc_n": {
            "metrics": metrics_hysoc_n,
            "retention": hysoc_retention,
            "diagnostics": hysoc_diag,
        },
    }

    diagnostics_path = os.path.join(out_dir, "diagnostics.json")
    with open(diagnostics_path, "w", newline="") as f:
        json.dump(diagnostics, f, indent=2)

    plot_pipeline_breakdown(
        out_path=os.path.join(out_dir, "pipeline_breakdown.png"),
        oracle_stages_s=oracle_stages_s,
        hysoc_diag=hysoc_diag,
    )
    plot_retention_vs_sed(
        out_path=os.path.join(out_dir, "retention_vs_sed.png"),
        oracle_metrics=diagnostics["oracle_n"],
        hysoc_metrics=diagnostics["hysoc_n"],
    )
    if not args.no_overlay_plot:
        plot_trajectory_overlay(
            out_path=os.path.join(out_dir, "trajectory_overlay.png"),
            raw_points=raw_points,
            oracle_items=processed_oracle_n,
            hysoc_items=hysoc_items,
        )

    print(
        "Oracle-N: "
        f"CR={metrics_oracle_n['cr']:.2f}, MeanSED={metrics_oracle_n['avg_sed_m']:.2f}m, "
        f"Latency={metrics_oracle_n['latency_us_per_point']:.1f}us/pt"
    )
    print(
        "HYSOC-N:  "
        f"CR={metrics_hysoc_n['cr']:.2f}, MeanSED={metrics_hysoc_n['avg_sed_m']:.2f}m, "
        f"Latency={metrics_hysoc_n['latency_us_per_point']:.1f}us/pt"
    )
    print(f"Saved diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
