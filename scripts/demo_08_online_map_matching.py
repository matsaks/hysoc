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

def plot_matching_results(raw_points, matched_points, G, output_img, input_filename):
    """Generates a side-by-side plot of raw GPS vs matched road sequences."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found. Skipping plot.")
        return

    # Extract geometries from the graph for the matched road IDs
    # To properly visualize the snapped route, we look up the geometry of each matched road_id
    matched_geoms = []
    
    # We create a mapping of osmid -> edge geometry 
    # to quickly find the road shape for plotting.
    from collections import defaultdict
    osmid_to_geom = defaultdict(list)
    for u, v, data in G.edges(data=True):
        if 'osmid' in data:
            # osmid can be a list or single value
            osmids = data['osmid'] if isinstance(data['osmid'], list) else [data['osmid']]
            for oid in osmids:
                key = str(oid)
                # If there's a geometry, use it. Otherwise create a straight line between u and v
                if 'geometry' in data:
                    osmid_to_geom[key].append(data['geometry'])
                else:
                    u_node = G.nodes[u]
                    v_node = G.nodes[v]
                    osmid_to_geom[key].append(LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])]))
                    
    # Only keep the unique consecutive road IDs to draw the path once per segment
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
        
        # Plot continuous thin line between raw points
        if len(raw_points) >= 2:
            raw_line = LineString([(p.lon, p.lat) for p in raw_points])
            gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax1, color='red', linewidth=1, alpha=0.3)
            
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Unmatched GPS Trajectory ({input_filename})")
        ax1.legend()
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Matched Edges ---
    print("Plotting map matched path...")
    import matplotlib.lines as mlines
    
    try:
        # Plot underlying graph nodes faintly for context
        nodes_data = [{'geometry': ShapelyPoint(data['x'], data['y'])} for _, data in G.nodes(data=True)]
        if nodes_data:
            gdf_nodes = gpd.GeoDataFrame(nodes_data, crs="EPSG:4326").to_crs(epsg=3857)
            gdf_nodes.plot(ax=ax2, color='gray', markersize=2, alpha=0.3)
            gdf_nodes.plot(ax=ax3, color='gray', markersize=2, alpha=0.3)
            
        if matched_geoms:
            gdf_edges = gpd.GeoDataFrame([{"geometry": geom} for geom in matched_geoms], crs="EPSG:4326")
            gdf_web_edges = gdf_edges.to_crs(epsg=3857)
            gdf_web_edges.plot(ax=ax2, color='blue', linewidth=4, alpha=0.7, label='Snapped Route (HMM)')
        else:
            print("Warning: No matching geometries found in graph to plot.")

        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
        ax2.set_axis_off()
        ax2.set_title(f"HMM Map Matched Sub-trajectory ({input_filename})")
        
        blue_line = mlines.Line2D([], [], color='blue', linewidth=4, label='HMM Inferred Route')
        gray_dot = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=4, label='OSM Node')
        ax2.legend(handles=[blue_line, gray_dot])
    except Exception as e:
        print(f"Error plotting plot 2: {e}")

    # --- Plot 3: Snapped Points ---
    print("Plotting snapped output points...")
    try:
        if matched_points:
            # Plot the actual output point positions to visually verify they snapped correctly
            matched_pts_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in matched_points]
            gpd.GeoDataFrame(matched_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax3, color='lime', markersize=15, alpha=0.9, label='Snapped GPS Fixes'
            )

        ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron)
        ax3.set_axis_off()
        ax3.set_title(f"Snapped Output Coordinates ({input_filename})")
        
        lime_dot = mlines.Line2D([], [], color='lime', marker='o', linestyle='None', markersize=6, label='Snapped Output Fix')
        ax3.legend(handles=[lime_dot, gray_dot])
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
        # default=os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv"),
        default=os.path.join(project_root, "data", "raw", "subset_50", "5013812.csv"),
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

    # Create output directory
    script_name = "demo_08_online_map_matching"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)

    # Save Mapping Output
    output_csv = os.path.join(output_dir, "matched_trajectory.csv")
    print(f"Saving matched trajectory to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id"])
        for p in matched_points:
            writer.writerow([
                p.lat, p.lon,
                p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id
            ])

    # Plot
    output_img = os.path.join(output_dir, "plot.png")
    plot_matching_results(raw_points, matched_points, G, output_img, input_filename)

if __name__ == "__main__":
    main()
