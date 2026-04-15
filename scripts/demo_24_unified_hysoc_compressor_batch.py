"""
Demo 24: Unified HYSOC Compressor (Batch)

Batch variant of demo_20_unified_hysoc_compressor.py.
Runs HYSOC compression for all CSV trajectories in an input folder and saves:
- Per-file compressed outputs and metrics.
- One aggregated metrics report over all successfully processed files.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from statistics import mean
from typing import Any

import osmnx as ox

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from core.point import Point
from core.stream import TrajectoryStream
from hysoc.hysocG import CompressionStrategy, HYSOCCompressor, HYSOCConfig

DEFAULT_INPUT_DIR: str = os.path.join("data", "raw", "London_Final_100")


def extract_compressed_points_separate(compressed_trajectory) -> tuple[list[Point], list[Point], list[list[Point]]]:
    """Extract compressed points as (all_points, stop_points, move_segments)."""
    all_points: list[Point] = []
    stop_points: list[Point] = []
    move_segments: list[list[Point]] = []

    for seg in compressed_trajectory.compressed_segments:
        if seg.segment_type == "stop":
            if hasattr(seg.compressed_data, "centroid"):
                stop_pt = seg.compressed_data.centroid
                all_points.append(stop_pt)
                stop_points.append(stop_pt)
        elif seg.segment_type == "move":
            data = seg.compressed_data
            move_pts: list[Point] = []
            if isinstance(data, list):
                move_pts = data
            elif isinstance(data, dict):
                move_pts = data.get("retained_points", [])
            elif hasattr(data, "points"):
                move_pts = data.points

            if move_pts:
                move_segments.append(move_pts)
                all_points.extend(move_pts)

    return all_points, stop_points, move_segments


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _summarize_strategy(per_file_results: list[dict[str, Any]], strategy_key: str) -> dict[str, Any]:
    strategy_rows = [r["strategies"][strategy_key] for r in per_file_results if strategy_key in r.get("strategies", {})]
    if not strategy_rows:
        return {"files": 0}

    ratios = [row["compression_ratio"] for row in strategy_rows]
    total_original = sum(row["original_points"] for row in strategy_rows)
    total_compressed = sum(row["compressed_points"] for row in strategy_rows)
    total_segments = sum(row["num_segments"] for row in strategy_rows)
    total_stops = sum(row["num_stops"] for row in strategy_rows)
    total_moves = sum(row["num_moves"] for row in strategy_rows)

    return {
        "files": len(strategy_rows),
        "compression_ratio": {
            "mean": mean(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "global": _safe_ratio(total_original, total_compressed),
        },
        "totals": {
            "original_points": total_original,
            "compressed_points": total_compressed,
            "num_segments": total_segments,
            "num_stops": total_stops,
            "num_moves": total_moves,
        },
    }


def process_single_file(data_path: str, strategy: str, output_dir: str) -> dict[str, Any]:
    """Run one file through the same HYSOC strategy flow as demo_20."""
    print(f"\nProcessing {os.path.basename(data_path)}")
    stream = TrajectoryStream(
        filepath=data_path,
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
    )
    raw_points = list(stream.stream())
    if not raw_points:
        raise ValueError("Input trajectory is empty.")

    lats = [p.lat for p in raw_points]
    lons = [p.lon for p in raw_points]
    north, south = max(lats) + 0.01, min(lats) - 0.01
    east, west = max(lons) + 0.01, min(lons) - 0.01

    print(
        f"  Downloading graph bbox W:{west:.4f} S:{south:.4f} E:{east:.4f} N:{north:.4f}"
    )
    graph = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    print(f"  Graph ready. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    results: dict[str, Any] = {
        "input_file": os.path.basename(data_path),
        "original_points": len(raw_points),
        "strategies": {},
    }

    if strategy in ["geometric", "both"]:
        config_g = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.GEOMETRIC,
            stop_max_eps_meters=25.0,
            stop_min_duration_seconds=30.0,
            dp_epsilon_meters=DP_DEFAULT_EPSILON_METERS,
        )
        compressed_g = HYSOCCompressor(config=config_g).compress(raw_points)
        all_points_g, stop_points_g, _move_segments_g = extract_compressed_points_separate(compressed_g)
        stops_count = sum(1 for s in compressed_g.compressed_segments if s.segment_type == "stop")
        moves_count = sum(1 for s in compressed_g.compressed_segments if s.segment_type == "move")
        factor_g = _safe_ratio(compressed_g.total_original_points, compressed_g.total_compressed_points)

        results["strategies"]["geometric"] = {
            "original_points": compressed_g.total_original_points,
            "compressed_points": compressed_g.total_compressed_points,
            "compression_ratio": factor_g,
            "num_segments": len(compressed_g.compressed_segments),
            "num_stops": stops_count,
            "num_moves": moves_count,
            "retained_stop_points": len(stop_points_g),
        }

        geom_csv = os.path.join(output_dir, "hysoc_g_compressed.csv")
        with open(geom_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "time"])
            for p in all_points_g:
                writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])

    if strategy in ["network_semantic", "both"]:
        config_n = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC,
            stop_max_eps_meters=25.0,
            stop_min_duration_seconds=30.0,
            osm_graph=graph,
            enable_map_matching=True,
        )
        compressed_n = HYSOCCompressor(config=config_n).compress(raw_points)
        all_points_n, stop_points_n, _move_segments_n = extract_compressed_points_separate(compressed_n)
        stops_count = sum(1 for s in compressed_n.compressed_segments if s.segment_type == "stop")
        moves_count = sum(1 for s in compressed_n.compressed_segments if s.segment_type == "move")
        factor_n = _safe_ratio(compressed_n.total_original_points, compressed_n.total_compressed_points)

        results["strategies"]["network_semantic"] = {
            "original_points": compressed_n.total_original_points,
            "compressed_points": compressed_n.total_compressed_points,
            "compression_ratio": factor_n,
            "num_segments": len(compressed_n.compressed_segments),
            "num_stops": stops_count,
            "num_moves": moves_count,
            "retained_stop_points": len(stop_points_n),
        }

        net_csv = os.path.join(output_dir, "hysoc_n_compressed.csv")
        with open(net_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "time"])
            for p in all_points_n:
                writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def write_aggregated_csv(output_dir: str, per_file_results: list[dict[str, Any]]) -> None:
    """Write a flat CSV with one row per input file and strategy."""
    csv_path = os.path.join(output_dir, "aggregated_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input_file",
                "strategy",
                "original_points",
                "compressed_points",
                "compression_ratio",
                "num_segments",
                "num_stops",
                "num_moves",
            ]
        )
        for file_result in per_file_results:
            input_file = file_result["input_file"]
            for strategy_name, row in file_result.get("strategies", {}).items():
                writer.writerow(
                    [
                        input_file,
                        strategy_name,
                        row["original_points"],
                        row["compressed_points"],
                        row["compression_ratio"],
                        row["num_segments"],
                        row["num_stops"],
                        row["num_moves"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 24: Unified HYSOC compressor for all files in a folder"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(project_root, DEFAULT_INPUT_DIR),
        help="Directory containing raw trajectory CSV files.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["geometric", "network_semantic", "both"],
        default="both",
        help="Compression strategy to test.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        sys.exit(1)

    csv_files = sorted(
        [
            os.path.join(input_dir, filename)
            for filename in os.listdir(input_dir)
            if filename.lower().endswith(".csv")
        ]
    )
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        sys.exit(1)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "data", "processed", script_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(csv_files)} files in {input_dir}")
    print(f"Writing outputs to {output_dir}")

    per_file_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, data_path in enumerate(csv_files, start=1):
        input_filename = os.path.splitext(os.path.basename(data_path))[0]
        file_output_dir = os.path.join(output_dir, input_filename)
        os.makedirs(file_output_dir, exist_ok=True)
        print(f"\n[{index}/{len(csv_files)}] {os.path.basename(data_path)}")

        try:
            file_result = process_single_file(data_path, args.strategy, file_output_dir)
            per_file_results.append(file_result)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed: {exc}")
            failures.append({"input_file": os.path.basename(data_path), "error": str(exc)})

    aggregated = {
        "timestamp": timestamp,
        "input_dir": input_dir,
        "strategy": args.strategy,
        "total_files_found": len(csv_files),
        "files_processed": len(per_file_results),
        "files_failed": len(failures),
        "failed_files": failures,
        "aggregated_by_strategy": {
            "geometric": _summarize_strategy(per_file_results, "geometric"),
            "network_semantic": _summarize_strategy(per_file_results, "network_semantic"),
        },
        "per_file": per_file_results,
    }

    aggregated_json_path = os.path.join(output_dir, "aggregated_metrics.json")
    with open(aggregated_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    write_aggregated_csv(output_dir, per_file_results)

    print("\n" + "=" * 60)
    print("Demo 24 completed.")
    print(f"Results directory: {output_dir}")
    print(f"Aggregated metrics: {aggregated_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
