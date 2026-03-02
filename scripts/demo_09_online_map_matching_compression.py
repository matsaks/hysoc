import os
import sys
import csv
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import osmnx as ox

# Add project root to sys.path to find benchmarks and src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.stream import TrajectoryStream
from hysoc.modules.map_matching.matcher import OnlineMapMatcher
from hysoc.modules.map_matching.wrapper import MapMatchedStreamWrapper

def plot_matching_results(raw_points, matched_points, compressed_points, G, output_img, input_filename):
    """Generates a side-by-side plot of raw GPS vs matched vs compressed sequences."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found. Skipping plot.")
        return

    # Extract geometries from the graph for the matched road IDs
    matched_geoms = []
    
    from collections import defaultdict
    osmid_to_geom = defaultdict(list)
    for u, v, data in G.edges(data=True):
        if 'osmid' in data:
            osmids = data['osmid'] if isinstance(data['osmid'], list) else [data['osmid']]
            for oid in osmids:
                key = str(oid)
                if 'geometry' in data:
                    osmid_to_geom[key].append(data['geometry'])
                else:
                    u_node = G.nodes[u]
                    v_node = G.nodes[v]
                    osmid_to_geom[key].append(LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])]))
                    
    consecutive_roads = []
    for p in matched_points:
        if p.road_id and (not consecutive_roads or consecutive_roads[-1] != p.road_id):
            consecutive_roads.append(p.road_id)

    for rid in consecutive_roads:
        if rid in osmid_to_geom:
            matched_geoms.extend(osmid_to_geom[rid])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))

    # --- Plot 1: Raw Points ---
    print("Plotting raw points...")
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in raw_points]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(ax=ax1, color='red', markersize=15, alpha=0.8, label='Raw GPS Fixes')
        
        if len(raw_points) >= 2:
            raw_line = LineString([(p.lon, p.lat) for p in raw_points])
            gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax1, color='red', linewidth=1, alpha=0.3)
            
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Unmatched GPS Trajectory ({len(raw_points)} points)")
        ax1.legend()
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Matched Edges & Points ---
    print("Plotting map matched path...")
    import matplotlib.lines as mlines
    
    try:
        nodes_data = [{'geometry': ShapelyPoint(data['x'], data['y'])} for _, data in G.nodes(data=True)]
        if nodes_data:
            gdf_nodes = gpd.GeoDataFrame(nodes_data, crs="EPSG:4326").to_crs(epsg=3857)
            gdf_nodes.plot(ax=ax2, color='gray', markersize=2, alpha=0.3)
            
        if matched_geoms:
            gdf_edges = gpd.GeoDataFrame([{"geometry": geom} for geom in matched_geoms], crs="EPSG:4326")
            gdf_web_edges = gdf_edges.to_crs(epsg=3857)
            gdf_web_edges.plot(ax=ax2, color='blue', linewidth=4, alpha=0.7, label='Snapped Route (HMM)')

        if matched_points:
            matched_pts_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in matched_points]
            gpd.GeoDataFrame(matched_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax2, color='lime', markersize=15, alpha=0.9, label='Snapped GPS Fixes'
            )

        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
        ax2.set_axis_off()
        ax2.set_title(f"Map Matched Output ({len(matched_points)} points)")
        
        blue_line = mlines.Line2D([], [], color='blue', linewidth=4, label='HMM Inferred Route')
        lime_dot = mlines.Line2D([], [], color='lime', marker='o', linestyle='None', markersize=6, label='Snapped Fix')
        gray_dot = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=4, label='OSM Node')
        ax2.legend(handles=[blue_line, lime_dot, gray_dot])
    except Exception as e:
        print(f"Error plotting plot 2: {e}")

    # --- Plot 3: Compressed Snapped Points ---
    print("Plotting compressed trajectory...")
    try:
        # Faintly show underlying network
        if nodes_data:
            gdf_nodes.plot(ax=ax3, color='gray', markersize=2, alpha=0.1)
        if matched_geoms:
            gdf_web_edges.plot(ax=ax3, color='blue', linewidth=2, alpha=0.1)
            
        if compressed_points:
            comp_pts_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in compressed_points]
            gpd.GeoDataFrame(comp_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax3, color='orange', markersize=35, alpha=0.9, label='Compressed GPS Fixes'
            )
            # Connect the compressed sequence
            if len(compressed_points) >= 2:
                comp_line = LineString([(p.lon, p.lat) for p in compressed_points])
                gpd.GeoDataFrame([{"geometry": comp_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax3, color='orange', linewidth=2, alpha=0.5)

        ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron)
        ax3.set_axis_off()
        ax3.set_title(f"Compressed Trajectory ({len(compressed_points)} points)")
        
        orange_dot = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=8, label='Compressed Semantic Fix')
        ax3.legend(handles=[orange_dot])
    except Exception as e:
        print(f"Error plotting plot 3: {e}")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Online HMM Map Matching.")
    parser.add_argument(
        "--input", 
        type=str, 
        default=os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv"),
        # default=os.path.join(project_root, "data", "raw", "subset_50", "5013812.csv"),
        help="Path to the raw trajectory CSV file."
    )
    args = parser.parse_args()
    data_path = args.input

    if not os.path.exists(data_path):
        print(f"Error: Input file {data_path} not found.")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    stream = TrajectoryStream(
        filepath=data_path,
        col_mapping={
            'lat': 'latitude',  # Must use raw coordinates 
            'lon': 'longitude',
            'timestamp': 'time'
        }
    )
    
    # We buffer all points just to calculate the bounding box and pass to the wrapper iterator
    raw_points = list(stream.stream())
    print(f"Loaded {len(raw_points)} points.")
    if not raw_points:
        return
        
    lats = [p.lat for p in raw_points]
    lons = [p.lon for p in raw_points]
    north, south = max(lats) + 0.01, min(lats) - 0.01
    east, west = max(lons) + 0.01, min(lons) - 0.01
    
    print(f"Downloading street graph for bounding box: W:{west:.4f}, S:{south:.4f}, E:{east:.4f}, N:{north:.4f}...")
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
    print(f"Graph downloaded. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    
    matcher = OnlineMapMatcher(
        G=G, 
        window_size=15, 
        max_dist=50, 
        max_dist_init=100, 
        min_prob_norm=0.001
    )
    
    # We need an iterator for the wrapper, recreating stream to be safe
    stream_again = TrajectoryStream(
        filepath=data_path,
        col_mapping={'lat': 'latitude', 'lon': 'longitude', 'timestamp': 'time'}
    )
    wrapper = MapMatchedStreamWrapper(point_stream=stream_again.stream(), matcher=matcher)
    
    print("Running MapMatchedStreamWrapper HMM Matching (this may take a moment)...")
    matched_points = list(wrapper.stream())
    
    # --- Compression Logic (STC-style on Road IDs) ---
    print(f"Compressing trajectory from {len(matched_points)} snapped points...")
    compressed_points = []
    current_road = None
    for i, p in enumerate(matched_points):
        # Always keep first and last points of the entire trajectory
        if i == 0 or i == len(matched_points) - 1:
            if not compressed_points or compressed_points[-1] != p:
                compressed_points.append(p)
            current_road = p.road_id
            continue
        
        # Keep points where the semantic mobility channel (road_id) changes
        if p.road_id != current_road:
            compressed_points.append(p)
            current_road = p.road_id
            
    print(f"Compressed trajectory to {len(compressed_points)} points. (Ratio: {len(compressed_points)/len(matched_points):.3f})")

    # Create output directory
    script_name = "demo_09_online_map_matching_compression"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)

    # Save Mapping Output
    output_csv = os.path.join(output_dir, "matched_trajectory.csv")
    print(f"Saving fully matched trajectory to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id"])
        for p in matched_points:
            writer.writerow([
                p.lat, p.lon,
                p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id
            ])

    # Save Compressed Output
    comp_csv = os.path.join(output_dir, "compressed_trajectory.csv")
    print(f"Saving compressed trajectory to {comp_csv}...")
    with open(comp_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id"])
        for p in compressed_points:
            writer.writerow([
                p.lat, p.lon,
                p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id
            ])

    import json
    metrics = {
        "original_points": len(raw_points),
        "matched_points": len(matched_points), 
        "compressed_points": len(compressed_points),
        "compression_ratio": round(len(raw_points) / len(compressed_points), 2) if len(compressed_points) else 0,
        "compression_ratio_matched": round(len(matched_points) / len(compressed_points), 2) if len(compressed_points) else 0
    }
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")

    # Plot
    output_img = os.path.join(output_dir, "plot.png")
    plot_matching_results(raw_points, matched_points, compressed_points, G, output_img, input_filename)

if __name__ == "__main__":
    main()
