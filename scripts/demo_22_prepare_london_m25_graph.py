# ruff: noqa: E402

"""
Demo 22: Prepare and cache OSM graph for London M25 area.

Purpose:
- Download (or reuse) one city-area graph for HYSOC-N/TRACE experiments.
- Persist graph to disk as GraphML.
- Report storage footprint and basic graph metadata.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_22_prepare_london_m25_graph")
DEFAULT_GRAPH_CACHE_DIR = os.path.join("data", "processed", "osm_graphs")
DEFAULT_GRAPH_CACHE_KEY = "london_m25_drive"

# London Bounding Box (M25 area)
DEFAULT_LAT_MIN = 51.28
DEFAULT_LAT_MAX = 51.69
DEFAULT_LON_MIN = -0.51
DEFAULT_LON_MAX = 0.33


def _import_osmnx():
    try:
        import osmnx as ox  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "osmnx is required for this demo. Install it with `pip install osmnx`."
        ) from exc
    return ox


def _resolve_paths(graph_cache_dir: str, graph_cache_key: str, output_root: str) -> Tuple[str, str]:
    cache_dir_abs = os.path.join(project_root, graph_cache_dir)
    os.makedirs(cache_dir_abs, exist_ok=True)
    graph_path = os.path.join(cache_dir_abs, f"{graph_cache_key}.graphml")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, output_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    return graph_path, out_dir


def _prepare_graph(
    *,
    graph_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    network_type: str,
    simplify: bool,
    refresh_graph: bool,
) -> Tuple[Any, Dict[str, Any]]:
    ox = _import_osmnx()

    metadata: Dict[str, Any] = {
        "graph_source": "cache_hit",
        "graph_download_time_s": None,
        "graph_load_time_s": None,
    }

    if os.path.exists(graph_path) and not refresh_graph:
        t0 = time.perf_counter()
        graph = ox.load_graphml(graph_path)
        t1 = time.perf_counter()
        metadata["graph_load_time_s"] = float(t1 - t0)
    else:
        t0 = time.perf_counter()
        graph = ox.graph_from_bbox(
            bbox=(lon_min, lat_min, lon_max, lat_max),
            network_type=network_type,
            simplify=simplify,
        )
        t1 = time.perf_counter()
        metadata["graph_source"] = "downloaded"
        metadata["graph_download_time_s"] = float(t1 - t0)
        ox.save_graphml(graph, graph_path)

    return graph, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 22: Download/cache London M25 OSM graph and measure storage footprint."
    )
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--network-type", type=str, default="drive")
    parser.add_argument("--no-simplify", action="store_true", help="Disable OSMnx topology simplification.")
    parser.add_argument("--graph-cache-dir", default=DEFAULT_GRAPH_CACHE_DIR)
    parser.add_argument("--graph-cache-key", default=DEFAULT_GRAPH_CACHE_KEY)
    parser.add_argument("--refresh-graph", action="store_true", help="Force re-download even if cache exists.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.lat_min >= args.lat_max:
        raise ValueError("--lat-min must be < --lat-max")
    if args.lon_min >= args.lon_max:
        raise ValueError("--lon-min must be < --lon-max")

    graph_path, out_dir = _resolve_paths(
        graph_cache_dir=args.graph_cache_dir,
        graph_cache_key=args.graph_cache_key,
        output_root=args.output_root,
    )

    print("Preparing London M25 graph cache...")
    print(f"Cache path: {graph_path}")
    print(
        "BBox: "
        f"lat[{args.lat_min}, {args.lat_max}] "
        f"lon[{args.lon_min}, {args.lon_max}]"
    )

    graph, prepare_meta = _prepare_graph(
        graph_path=graph_path,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        network_type=args.network_type,
        simplify=not args.no_simplify,
        refresh_graph=args.refresh_graph,
    )

    graph_size_bytes = os.path.getsize(graph_path)
    graph_size_mb = graph_size_bytes / (1024 * 1024)
    n_nodes = int(len(graph.nodes))
    n_edges = int(len(graph.edges))

    results = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "graph_file_path": graph_path,
        "graph_file_size_bytes": int(graph_size_bytes),
        "graph_file_size_mb": float(graph_size_mb),
        "graph_nodes": n_nodes,
        "graph_edges": n_edges,
        "network_type": args.network_type,
        "simplify": bool(not args.no_simplify),
        "bbox": {
            "lat_min": float(args.lat_min),
            "lat_max": float(args.lat_max),
            "lon_min": float(args.lon_min),
            "lon_max": float(args.lon_max),
        },
        "graph_cache_key": args.graph_cache_key,
        "graph_source": prepare_meta["graph_source"],
        "graph_download_time_s": prepare_meta["graph_download_time_s"],
        "graph_load_time_s": prepare_meta["graph_load_time_s"],
    }

    meta_json_path = os.path.join(out_dir, "graph_cache_metadata.json")
    with open(meta_json_path, "w", newline="") as f:
        json.dump(results, f, indent=2)

    print("\nGraph ready.")
    print(f"- Source: {results['graph_source']}")
    print(f"- Nodes: {results['graph_nodes']}")
    print(f"- Edges: {results['graph_edges']}")
    print(f"- File size: {results['graph_file_size_mb']:.2f} MB ({results['graph_file_size_bytes']} bytes)")
    if results["graph_download_time_s"] is not None:
        print(f"- Download time: {results['graph_download_time_s']:.2f} s")
    if results["graph_load_time_s"] is not None:
        print(f"- Load time: {results['graph_load_time_s']:.2f} s")
    print(f"- Metadata JSON: {meta_json_path}")


if __name__ == "__main__":
    main()

