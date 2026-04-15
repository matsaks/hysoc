"""
Demo: Unified HYSOC Compressor with Map Matching

This demo tests the unified HYSOCCompressor that orchestrates:
- Module I: STEP Segmentation
- Module II: Stop Compression
- Module III: Move Compression (Geometric or Network-Semantic)

Demonstrates both compression strategies:
1. HYSOC-G: Geometric move compression (SquishCompressor)
2. HYSOC-N: Network-Semantic move compression (TraceCompressor with map matching)

Uses the same map and data as demo_09_online_map_matching_compression.py
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import osmnx as ox

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from core.stream import TrajectoryStream
from core.point import Point
from hysoc.hysocG import HYSOCCompressor, HYSOCConfig, CompressionStrategy

# Default demo input file (can be overridden with --input)
DEFAULT_INPUT_FILE: str = os.path.join("data", "raw", "subset_50", "4494499.csv")



def plot_compression_results(
    raw_points: List[Point],
    compressed_data_g: dict,
    compressed_data_n: dict,
    G,
    output_dir: str,
    input_filename: str,
):
    """
    Generates a 2x2 plot comparing:
    - Top-left: Raw GPS points
    - Top-right: HYSOC-G (Geometric) compression
    - Bottom-left: HYSOC-N (Network-Semantic) compression
    - Bottom-right: Both strategies overlaid
    
    Stops are highlighted as large red dots, moves in strategy colors.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
        import matplotlib.lines as mlines
    except ImportError:
        print("geopandas, shapely, or contextily not found. Skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    ax_raw = axes[0, 0]
    ax_geom = axes[0, 1]
    ax_net = axes[1, 0]
    ax_both = axes[1, 1]

    # --- Plot 1: Raw Points ---
    print("Plotting raw trajectory...")
    try:
        raw_pts_data = [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in raw_points]
        gdf_raw = gpd.GeoDataFrame(raw_pts_data, crs="EPSG:4326")
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(
            ax=ax_raw, color="blue", markersize=8, alpha=0.5, label="Raw GPS Points"
        )

        if len(raw_points) >= 2:
            raw_line = LineString([(p.lon, p.lat) for p in raw_points])
            gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326").to_crs(
                epsg=3857
            ).plot(ax=ax_raw, color="blue", linewidth=1, alpha=0.2)

        ctx.add_basemap(ax_raw, source=ctx.providers.CartoDB.Positron)
        ax_raw.set_axis_off()
        ax_raw.set_title(
            f"Raw Trajectory ({len(raw_points)} points)", fontsize=14, fontweight="bold"
        )
        ax_raw.legend()
    except Exception as e:
        print(f"Error plotting raw trajectory: {e}")

    # --- Plot 2: HYSOC-G (Geometric) ---
    print("Plotting HYSOC-G (Geometric)...")
    try:
        geom_points = compressed_data_g.get("reconstructed_points", compressed_data_g["compressed_points"])
        geom_stops = compressed_data_g.get("stops", [])
        geom_move_segments = compressed_data_g.get("move_segments", [])
        
        # Count total move points across all segments
        total_move_points = sum(len(seg) for seg in geom_move_segments)
        
        # Plot continuous trajectory
        if len(geom_points) >= 2:
            geom_pts_gdf = gpd.GeoDataFrame(
                [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in geom_points],
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            coords = [(geom.x, geom.y) for geom in geom_pts_gdf.geometry]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax_geom.plot(xs, ys, color="blue", linewidth=2, alpha=0.6, zorder=2)
            geom_pts_gdf.plot(ax=ax_geom, color="blue", markersize=20, alpha=0.5, zorder=2)
        
        # Plot stops on top (larger, red) - these are the KEY points
        if geom_stops:
            stop_pts_data = [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in geom_stops]
            gpd.GeoDataFrame(stop_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax_geom, color="red", markersize=150, alpha=0.9, label=f"Stops ({len(geom_stops)})", marker='o', edgecolor='darkred', linewidth=2
            )

        ctx.add_basemap(ax_geom, source=ctx.providers.CartoDB.Positron)
        ax_geom.set_axis_off()
        total_orig = len(raw_points)
        total_comp_g = len(compressed_data_g["compressed_points"])
        factor_g = total_orig / total_comp_g if total_comp_g > 0 else 0
        ax_geom.set_title(
            f"HYSOC-G: Geometric\n{len(geom_stops)} stops + {total_move_points} move points → {factor_g:.1f}x compression",
            fontsize=12,
            fontweight="bold",
        )
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label=f'Move Points ({total_move_points})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label=f'Stops ({len(geom_stops)})')
        ]
        ax_geom.legend(handles=legend_elements, loc='upper left')
    except Exception as e:
        print(f"Error plotting HYSOC-G: {e}")

    # --- Plot 3: HYSOC-N (Network-Semantic) ---
    print("Plotting HYSOC-N (Network-Semantic)...")
    try:
        net_points = compressed_data_n.get("reconstructed_points", compressed_data_n["compressed_points"])
        net_stops = compressed_data_n.get("stops", [])
        net_move_segments = compressed_data_n.get("move_segments", [])
        
        # Count total move points across all segments
        total_move_points = sum(len(seg) for seg in net_move_segments)
        
        # Plot continuous trajectory
        if len(net_points) >= 2:
            net_pts_gdf = gpd.GeoDataFrame(
                [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in net_points],
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            coords = [(geom.x, geom.y) for geom in net_pts_gdf.geometry]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax_net.plot(xs, ys, color="orange", linewidth=2, alpha=0.6, zorder=2)
            net_pts_gdf.plot(ax=ax_net, color="orange", markersize=20, alpha=0.5, zorder=2)
        
        # Plot stops on top (larger, red) - these are the KEY points
        if net_stops:
            stop_pts_data = [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in net_stops]
            gpd.GeoDataFrame(stop_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax_net, color="red", markersize=150, alpha=0.9, label=f"Stops ({len(net_stops)})", marker='o', edgecolor='darkred', linewidth=2
            )

        ctx.add_basemap(ax_net, source=ctx.providers.CartoDB.Positron)
        ax_net.set_axis_off()
        total_orig = len(raw_points)
        total_comp_n = len(compressed_data_n["compressed_points"])
        factor_n = total_orig / total_comp_n if total_comp_n > 0 else 0
        ax_net.set_title(
            f"HYSOC-N: Network-Semantic\n{len(net_stops)} stops + {total_move_points} move points → {factor_n:.1f}x compression",
            fontsize=12,
            fontweight="bold",
        )
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label=f'Move Points ({total_move_points})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label=f'Stops ({len(net_stops)})')
        ]
        ax_net.legend(handles=legend_elements, loc='upper left')
    except Exception as e:
        print(f"Error plotting HYSOC-N: {e}")

    # --- Plot 4: Both Overlaid ---
    print("Plotting overlay comparison...")
    try:
        # Faint background of raw trajectory
        raw_pts_data = [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in raw_points]
        gdf_raw = gpd.GeoDataFrame(raw_pts_data, crs="EPSG:4326")
        gdf_raw.to_crs(epsg=3857).plot(
            ax=ax_both, color="lightgray", markersize=2, alpha=0.2, label="Raw"
        )

        # Geometric continuous trajectory
        geom_points = compressed_data_g.get("reconstructed_points", [])
        if len(geom_points) >= 2:
            geom_pts_gdf = gpd.GeoDataFrame(
                [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in geom_points],
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            coords = [(geom.x, geom.y) for geom in geom_pts_gdf.geometry]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax_both.plot(xs, ys, color="blue", linewidth=1.5, alpha=0.5, zorder=2)
            geom_pts_gdf.plot(ax=ax_both, color="blue", markersize=10, alpha=0.4, zorder=2)

        # Network continuous trajectory
        net_points = compressed_data_n.get("reconstructed_points", [])
        if len(net_points) >= 2:
            net_pts_gdf = gpd.GeoDataFrame(
                [{"geometry": ShapelyPoint(p.lon, p.lat)} for p in net_points],
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            coords = [(geom.x, geom.y) for geom in net_pts_gdf.geometry]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax_both.plot(xs, ys, color="orange", linewidth=1.5, alpha=0.5, zorder=2)
            net_pts_gdf.plot(ax=ax_both, color="orange", markersize=10, alpha=0.4, zorder=2)
        
        # All stops overlaid - large red dots on top (these are identical for both strategies)
        all_stops = compressed_data_g.get("stops", [])
        if all_stops:
            stop_pts_data = [
                {"geometry": ShapelyPoint(p.lon, p.lat)} for p in all_stops
            ]
            gpd.GeoDataFrame(stop_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax_both, color="red", markersize=160, alpha=0.9, label=f"Stops ({len(all_stops)})", marker='o', edgecolor='darkred', linewidth=2
            )

        ctx.add_basemap(ax_both, source=ctx.providers.CartoDB.Positron)
        ax_both.set_axis_off()
        ax_both.set_title("Pipeline: Stops (red) + Moves (blue/orange)", fontsize=12, fontweight="bold")
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='HYSOC-G Moves'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='HYSOC-N Moves'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label=f'Stops ({len(all_stops)})')
        ]
        ax_both.legend(handles=legend_elements, loc='upper left')
    except Exception as e:
        print(f"Error plotting overlay: {e}")

    plt.tight_layout()
    output_img = os.path.join(output_dir, "hysoc_compression_comparison.png")
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_img}")
    plt.close()


def extract_compressed_points_separate(compressed_trajectory) -> Tuple[List[Point], List[Point], List[List[Point]]]:
    """
    Extract Point objects from the compressed trajectory, keeping move segments SEPARATE.
    Returns tuple of (all_points, stop_points, move_segments).
    Each element of move_segments is a list of points from one move segment.
    This preserves segment boundaries to prevent lines crossing stop points.
    Works for both HYSOC-G and HYSOC-N strategies.
    """
    all_points = []
    stop_points = []
    move_segments = []  # List of move segment point lists (NOT flattened)
    
    for seg in compressed_trajectory.compressed_segments:
        if seg.segment_type == "stop":
            # Stop compression results in 1 centroid point
            if hasattr(seg.compressed_data, "centroid"):
                stop_pt = seg.compressed_data.centroid
                all_points.append(stop_pt)
                stop_points.append(stop_pt)
        elif seg.segment_type == "move":
            # For moves, extract point list and keep segment separate
            data = seg.compressed_data
            move_pts = []
            if isinstance(data, list):  # HYSOC-G returns List[Point]
                move_pts = data
            elif isinstance(data, dict):  # HYSOC-N returns {trace_result, retained_points}
                # Use the retained points extracted by TRACE compression logic
                move_pts = data.get('retained_points', [])
            elif hasattr(data, "points"):  # Move object
                move_pts = data.points
            
            if move_pts:
                move_segments.append(move_pts)
                all_points.extend(move_pts)
    
    return all_points, stop_points, move_segments


def extract_compressed_points(compressed_trajectory) -> Tuple[List[Point], List[Point], List[Point]]:
    """
    Extract actual Point objects from the compressed trajectory.
    Returns tuple of (all_points, stop_points, move_points).
    Works for both HYSOC-G and HYSOC-N strategies.
    """
    all_points = []
    stop_points = []
    move_points = []
    
    for seg in compressed_trajectory.compressed_segments:
        if seg.segment_type == "stop":
            # Stop compression results in 1 centroid point
            if hasattr(seg.compressed_data, "centroid"):
                stop_pt = seg.compressed_data.centroid
                all_points.append(stop_pt)
                stop_points.append(stop_pt)
        elif seg.segment_type == "move":
            # For moves, extract point list
            data = seg.compressed_data
            if isinstance(data, list):  # HYSOC-G returns List[Point]
                all_points.extend(data)
                move_points.extend(data)
            elif isinstance(data, dict):  # HYSOC-N returns {trace_result, retained_points}
                # Use the retained points extracted by TRACE compression logic
                retained = data.get('retained_points', [])
                all_points.extend(retained)
                move_points.extend(retained)
            elif hasattr(data, "points"):  # Move object
                all_points.extend(data.points)
                move_points.extend(data.points)
    
    return all_points, stop_points, move_points


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Unified HYSOC Compressor with two strategies"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(project_root, DEFAULT_INPUT_FILE),
        help="Path to the raw trajectory CSV file.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["geometric", "network_semantic", "both"],
        default="both",
        help="Compression strategy to test.",
    )
    args = parser.parse_args()

    data_path = args.input

    if not os.path.exists(data_path):
        print(f"Error: Input file {data_path} not found.")
        sys.exit(1)

    print(f"Loading trajectory from {data_path}...")
    stream = TrajectoryStream(
        filepath=data_path,
        col_mapping={
            "lat": "latitude",
            "lon": "longitude",
            "timestamp": "time",
        },
    )

    raw_points = list(stream.stream())
    print(f"Loaded {len(raw_points)} points.")
    if not raw_points:
        return

    # Get bounding box
    lats = [p.lat for p in raw_points]
    lons = [p.lon for p in raw_points]
    north, south = max(lats) + 0.01, min(lats) - 0.01
    east, west = max(lons) + 0.01, min(lons) - 0.01

    print(
        f"Downloading street graph for bounding box: W:{west:.4f}, S:{south:.4f}, "
        f"E:{east:.4f}, N:{north:.4f}..."
    )
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    print(f"Graph downloaded. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    # Create output directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(
        project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "input_file": os.path.basename(data_path),
        "original_points": len(raw_points),
        "timestamp": timestamp,
        "strategies": {},
    }

    # Store compressed data for plotting
    compressed_data_g = None
    compressed_data_n = None

    # Test HYSOC-G (Geometric)
    if args.strategy in ["geometric", "both"]:
        print("\n" + "=" * 60)
        print("Testing HYSOC-G: Geometric Move Compression")
        print("=" * 60)

        config_g = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.GEOMETRIC,
            stop_max_eps_meters=25.0,  # Decreased to consume fewer points into stops
            stop_min_duration_seconds=30.0,  # Increased waiting time required
            dp_epsilon_meters=15.0,  # DP max error tolerance in meters
        )

        compressor_g = HYSOCCompressor(config=config_g)
        print(compressor_g.get_compression_summary())

        compressed_g = compressor_g.compress(raw_points)
        all_points_g, stop_points_g, move_segments_g = extract_compressed_points_separate(compressed_g)

        print(f"\nSegment Breakdown:")
        stops_count = sum(1 for s in compressed_g.compressed_segments if s.segment_type == "stop")
        moves_count = sum(1 for s in compressed_g.compressed_segments if s.segment_type == "move")
        print(f"  Total segments: {len(compressed_g.compressed_segments)}")
        print(f"  Stop segments: {stops_count}")
        print(f"  Move segments: {moves_count}")
        
        for i, seg in enumerate(compressed_g.compressed_segments):
            seg_type = seg.segment_type.upper()
            orig_pts = len(seg.original_segment.points)
            if seg.segment_type == "stop":
                comp_pts = 1
                factor = orig_pts / comp_pts if comp_pts > 0 else 0
                print(f"    [{i}] {seg_type}: {orig_pts} points → 1 centroid (factor: {factor:.1f}x)")
            else:
                comp_pts = len(seg.compressed_data) if isinstance(seg.compressed_data, list) else len(seg.original_segment.points)
                factor = orig_pts / comp_pts if comp_pts > 0 else 0
                print(f"    [{i}] {seg_type}: {orig_pts} points → {comp_pts} points (factor: {factor:.1f}x)")

        print(f"\nOverall Compression:")
        print(f"  Original points: {compressed_g.total_original_points}")
        print(f"  Compressed points: {compressed_g.total_compressed_points}")
        factor_g = compressed_g.total_original_points / compressed_g.total_compressed_points if compressed_g.total_compressed_points > 0 else 0
        print(f"  Compression factor: {factor_g:.1f}x")

        results["strategies"]["geometric"] = {
            "original_points": compressed_g.total_original_points,
            "compressed_points": compressed_g.total_compressed_points,
            "compression_ratio": factor_g,
            "num_segments": len(compressed_g.compressed_segments),
            "num_stops": stops_count,
            "num_moves": moves_count,
        }

        # Save results
        geom_csv = os.path.join(output_dir, "hysoc_g_compressed.csv")
        print(f"\nSaving HYSOC-G results to {geom_csv}...")
        with open(geom_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "time"])
            for p in all_points_g:
                writer.writerow(
                    [p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")]
                )

        compressed_data_g = {
            "compressed_points": all_points_g,
            "compression_ratio": factor_g,
            "stops": stop_points_g,
            "move_segments": move_segments_g,  # Separate move segments (not flattened)
            "reconstructed_points": compressed_g.get_reconstructed_points(),
        }

    # Test HYSOC-N (Network-Semantic)
    if args.strategy in ["network_semantic", "both"]:
        print("\n" + "=" * 60)
        print("Testing HYSOC-N: Network-Semantic Move Compression")
        print("=" * 60)

        config_n = HYSOCConfig(
            move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC,
            stop_max_eps_meters=25.0,  # Decreased to consume fewer points into stops
            stop_min_duration_seconds=30.0,  # Increased waiting time required
            osm_graph=G,
            enable_map_matching=True,
        )

        compressor_n = HYSOCCompressor(config=config_n)
        print(compressor_n.get_compression_summary())

        print("Running compression (this may take a moment for map matching)...")
        compressed_n = compressor_n.compress(raw_points)
        all_points_n, stop_points_n, move_segments_n = extract_compressed_points_separate(compressed_n)

        print(f"\nSegment Breakdown:")
        stops_count = sum(1 for s in compressed_n.compressed_segments if s.segment_type == "stop")
        moves_count = sum(1 for s in compressed_n.compressed_segments if s.segment_type == "move")
        print(f"  Total segments: {len(compressed_n.compressed_segments)}")
        print(f"  Stop segments: {stops_count}")
        print(f"  Move segments: {moves_count}")
        
        for i, seg in enumerate(compressed_n.compressed_segments):
            seg_type = seg.segment_type.upper()
            orig_pts = len(seg.original_segment.points)
            if seg.segment_type == "stop":
                comp_pts = 1
                factor = orig_pts / comp_pts if comp_pts > 0 else 0
                print(f"    [{i}] {seg_type}: {orig_pts} points → 1 centroid (factor: {factor:.1f}x)")
            else:
                # For TRACE, count unique road IDs as the compressed representation
                unique_roads = len(set(p.road_id for p in seg.original_segment.points if p.road_id))
                comp_pts = unique_roads if unique_roads > 0 else 1
                factor = orig_pts / comp_pts if comp_pts > 0 else 0
                print(f"    [{i}] {seg_type}: {orig_pts} points → {unique_roads} roads (factor: {factor:.1f}x)")

        print(f"\nOverall Compression:")
        print(f"  Original points: {compressed_n.total_original_points}")
        print(f"  Compressed points: {compressed_n.total_compressed_points}")
        factor_n = compressed_n.total_original_points / compressed_n.total_compressed_points if compressed_n.total_compressed_points > 0 else 0
        print(f"  Compression factor: {factor_n:.1f}x")

        results["strategies"]["network_semantic"] = {
            "original_points": compressed_n.total_original_points,
            "compressed_points": compressed_n.total_compressed_points,
            "compression_ratio": factor_n,
            "num_segments": len(compressed_n.compressed_segments),
            "num_stops": stops_count,
            "num_moves": moves_count,
        }

        # Save results
        net_csv = os.path.join(output_dir, "hysoc_n_compressed.csv")
        print(f"\nSaving HYSOC-N results to {net_csv}...")
        with open(net_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "time"])
            for p in all_points_n:
                writer.writerow(
                    [p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")]
                )

        compressed_data_n = {
            "compressed_points": all_points_n,
            "compression_ratio": factor_n,
            "stops": stop_points_n,
            "move_segments": move_segments_n,  # Separate move segments (not flattened)
            "reconstructed_points": compressed_n.get_reconstructed_points(),
        }

    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    # Plot if both strategies were run
    if args.strategy == "both" and compressed_data_g and compressed_data_n:
        print("\nGenerating comparison plots...")
        plot_compression_results(
            raw_points, compressed_data_g, compressed_data_n, G, output_dir, input_filename
        )

    print("\n" + "=" * 60)
    print(f"Demo completed. Results saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
