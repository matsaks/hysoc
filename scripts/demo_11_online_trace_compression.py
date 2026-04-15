import os
import sys
import csv
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import math
import osmnx as ox

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from core.point import Point
from engines.move_compression.trace import TraceCompressor, TraceConfig

def load_trajectory(filepath: str) -> list[Point]:
    """Loads a trajectory from a CSV file."""
    points = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Support multiple column naming conventions
                time_str = row.get('time', row.get('timestamp'))
                lat_str = row.get('latitude', row.get('lat'))
                lon_str = row.get('longitude', row.get('lon'))
                
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                lat = float(lat_str)
                lon = float(lon_str)
                # road_id is optional and None by default
                points.append(Point(lat=lat, lon=lon, timestamp=dt, obj_id="demo_obj"))
            except (ValueError, TypeError) as e:
                continue
    return points

def extract_retained_points(points: list[Point], gamma: float) -> list[Point]:
    """
    Simulates the speed-based representation logic to identify which points
    are retained as key points in the compressed representation.
    """
    retained_points = []
    
    def lat_lon_dist(p1: Point, p2: Point) -> float:
        # Equirectangular approximation for speed
        R = 6371000.0
        lat1 = math.radians(p1.lat)
        lat2 = math.radians(p2.lat)
        dlat = lat2 - lat1
        dlon = math.radians(p2.lon - p1.lon)
        x = dlon * math.cos((lat1 + lat2) / 2.0)
        y = dlat
        return R * math.sqrt(x*x + y*y)

    current_road_id = None
    last_stored_speed = -1.0 

    for i, p in enumerate(points):
        # 1. Road Segment Change / Start
        if i == 0 or p.road_id != current_road_id:
            current_road_id = p.road_id
            retained_points.append(p)
            last_stored_speed = 0.0
            continue

        # 2. Same road segment
        prev_p = points[i-1]
        dist = lat_lon_dist(prev_p, p)
        time_diff = (p.timestamp - prev_p.timestamp).total_seconds()
        
        if time_diff > 0:
            current_speed = dist / time_diff
        else:
            current_speed = last_stored_speed 

        if abs(current_speed - last_stored_speed) > gamma:
            retained_points.append(p)
            last_stored_speed = current_speed
            
    # Always ensure the last point is included if not already
    if retained_points and retained_points[-1] != points[-1]:
       retained_points.append(points[-1])
            
    return retained_points

def plot_trace_results(raw_points, compressed_points, G, output_img):
    """Generates a side-by-side plot of raw GPS vs compressed sequence with matching map background."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found. Skipping plot.")
        return

    # Extract all edges from G to plot faintly in background (projected correctly)
    graph_edges_geoms = []
    if G:
        for u, v, data in G.edges(data=True):
            if 'geometry' in data:
                graph_edges_geoms.append(data['geometry'])
            else:
                # Create straight line if no geometry
                u_node = G.nodes[u]
                v_node = G.nodes[v]
                graph_edges_geoms.append(LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])]))
    
    gdf_graph_edges = None
    if graph_edges_geoms:
        gdf_graph_edges = gpd.GeoDataFrame([{'geometry': g} for g in graph_edges_geoms], crs="EPSG:4326").to_crs(epsg=3857)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # --- Plot 1: Raw Points ---
    print("Plotting raw points...")
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in raw_points]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        
        # Plot full graph faintly on raw plot too? Maybe just the trajectory.
        # User requested "map structured background". Tile map (CartoDB) provides this.
        
        gdf_raw_web.plot(ax=ax1, color='red', markersize=15, alpha=0.8, label='Raw GPS Fixes')
        
        if len(raw_points) >= 2:
            raw_line = LineString([(p.lon, p.lat) for p in raw_points])
            gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax1, color='red', linewidth=1, alpha=0.3)
            
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Original Trajectory ({len(raw_points)} points)")
        ax1.legend()
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Compressed Points ---
    print("Plotting compressed trajectory...")
    try:
        # 1. Plot background network (projected!)
        if gdf_graph_edges is not None:
            gdf_graph_edges.plot(ax=ax2, color='gray', linewidth=0.5, alpha=0.2)

        # 2. Plot Compressed Points
        if compressed_points:
            comp_pts_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in compressed_points]
            gpd.GeoDataFrame(comp_pts_data, crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax2, color='blue', markersize=35, alpha=0.9, label='TRACE Retained Points'
            )
            # Connect the compressed sequence
            if len(compressed_points) >= 2:
                comp_line = LineString([(p.lon, p.lat) for p in compressed_points])
                gpd.GeoDataFrame([{"geometry": comp_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax2, color='blue', linewidth=2, alpha=0.5)

        # 3. Add Basemap
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
        ax2.set_axis_off()
        ax2.set_title(f"Compressed Representation ({len(compressed_points)} points)")
        ax2.legend()
    except Exception as e:
        print(f"Error plotting plot 2: {e}")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

def main():
    parser = argparse.ArgumentParser(description="Demonstrate TRACE Algorithm Compression.")
    parser.add_argument(
        "--input", 
        type=str, 
        default=os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv"),
        help="Path to the raw trajectory CSV file."
    )
    args = parser.parse_args()
    data_path = args.input
    
    # Setup Output Directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(data_path)
    file_id = os.path.splitext(filename)[0]
    
    # Store under demo_10 specific folder
    output_dir = os.path.join(project_root, "data", "processed", "demo_10_online_trace_compression", f"{timestamp_str}_{file_id}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    trajectory = load_trajectory(data_path)
    
    if not trajectory:
        print("No trajectory data loaded.")
        return

    print(f"Loaded {len(trajectory)} points.")

    # 2. Get Map Data (for plotting context)
    lats = [p.lat for p in trajectory]
    lons = [p.lon for p in trajectory]
    north, south = max(lats) + 0.005, min(lats) - 0.005
    east, west = max(lons) + 0.005, min(lons) - 0.005
    
    print(f"Downloading street graph for bounding box (for visualization)...")
    try:
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
    except Exception as e:
        print(f"Could not download graph: {e}")
        G = None

    # 3. Configure TRACE
    # Reduced gamma to 1.5 to capture more corner points (speed changes)
    config = TraceConfig(gamma=3.8, epsilon=5.0, k=4) 
    compressor = TraceCompressor(config)

    # 4. Perform Compression
    print("Running TRACE compression...")
    # NOTE: This runs the algorithmic compression logic (Referential + Speed).
    compressed_data = compressor.compress(trajectory)
    
    # 5. Extract Retained Points for Visualization
    retained_points = extract_retained_points(trajectory, config.gamma)
    
    num_compressed_records = len(compressed_data['V']) # Count of V-packets (Speed updates / Matches)
    compression_ratio = len(trajectory) / max(1, num_compressed_records)
    
    print(f"Original Points: {len(trajectory)}")
    print(f"Retained Geometric Points: {len(retained_points)}")
    print(f"Compressed Records (Stream Size): {num_compressed_records}")
    print(f"Compression Ratio (Stream): {compression_ratio:.2f}")

    # 6. Save Stats and Compressed CSV
    comp_csv = os.path.join(output_dir, "compressed_trajectory.csv")
    with open(comp_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id"])
        for p in retained_points:
            writer.writerow([
                p.lat, p.lon,
                p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id
            ])

    import json
    metrics = {
        "original_points": len(trajectory),
        "retained_points": len(retained_points),
        "compressed_records": num_compressed_records,
        "compression_ratio": round(compression_ratio, 2)
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 7. Plot
    output_img_path = os.path.join(output_dir, "trace_compression_map.png")
    plot_trace_results(trajectory, retained_points, G, output_img_path)
    print("Done.")

if __name__ == "__main__":
    main()
