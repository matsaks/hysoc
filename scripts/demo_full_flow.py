
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import os
import sys

# Add project root to sys.path to find benchmarks and src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from hysoc.modules.stop_compression.compressor import StopCompressor, CompressedStop
from hysoc.modules.move_compression.squish import SquishCompressor

def load_trajectory(filepath: str) -> List[Point]:
    """
    Loads a trajectory from a CSV file.
    Assumes columns: time, latitude, longitude
    """
    points = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Format: 2022-11-09 17:32:43
                dt = datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                # Use a dummy obj_id for now
                points.append(Point(lat=lat, lon=lon, timestamp=dt, obj_id="demo_obj"))
            except ValueError as e:
                print(f"Skipping row due to error: {e}")
                continue
    return points

def main():
    # 1. Load Data
    # Construct absolute path to data file
    data_path = os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv")
    
    print(f"Loading data from {data_path}...")
    trajectory = load_trajectory(data_path)
    print(f"Loaded {len(trajectory)} points.")

    if not trajectory:
        print("No points loaded. Exiting.")
        return

    # 2. Segment
    print("Running STSSOracleSklearn...")
    # Parameters from notebook: min_samples=5, max_eps=10m, min_duration=10s
    oracle = STSSOracleSklearn(min_samples=5, max_eps=10, min_duration_seconds=10)
    segments = oracle.process(trajectory)
    print(f"Segmentation complete. Found {len(segments)} segments.")
    
    n_stops = sum(1 for s in segments if isinstance(s, Stop))
    n_moves = sum(1 for s in segments if isinstance(s, Move))
    print(f" - Stops: {n_stops}")
    print(f" - Moves: {n_moves}")

    # 3. Compress
    print("Compressing...")
    stop_compressor = StopCompressor()
    # Use Squish with capacity 50 for moves
    move_compressor = SquishCompressor(capacity=50)
    
    processed_items = []

    for seg in segments:
        if isinstance(seg, Stop):
            compressed_stop = stop_compressor.compress(seg.points)
            processed_items.append(compressed_stop)
        elif isinstance(seg, Move):
            # Dynamic capacity: keep 30% of points, but at least 10
            dynamic_capacity = max(10, int(len(seg.points) * 0.3))
            
            # Compress move points using Squish with dynamic capacity
            compressed_points = move_compressor.compress(seg.points, capacity=dynamic_capacity)
            # Create a simplified Move object with compressed points
            compressed_move = Move(points=compressed_points)
            processed_items.append(compressed_move)

    print(f"Compression complete. Result stream length: {len(processed_items)} items.")

    # 4. Save to CSV
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_csv = os.path.join(output_dir, f"full_flow_output_{input_filename}.csv")
    print(f"Saving to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "start_time", "end_time", "type"])
        
        for item in processed_items:
            if isinstance(item, CompressedStop):
                writer.writerow([
                    item.centroid.lat, item.centroid.lon,
                    item.start_time, item.end_time, "STOP"
                ])
            elif isinstance(item, Move):
                for p in item.points:
                     writer.writerow([
                        p.lat, p.lon,
                        p.timestamp, p.timestamp, "MOVE_POINT"
                    ])

    # 5. Visualize
    print("Generating visualization...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found. Please install them.")
        return

    # Create GDF data
    data = []
    for item in processed_items:
        if isinstance(item, Move):
            if not item.points:
                continue
            if len(item.points) < 2:
                geom = ShapelyPoint(item.points[0].lon, item.points[0].lat)
            else:
                coords = [(p.lon, p.lat) for p in item.points]
                geom = LineString(coords)
            data.append({"type": "Move", "geometry": geom})
        elif isinstance(item, CompressedStop):
            geom = ShapelyPoint(item.centroid.lon, item.centroid.lat)
            data.append({"type": "Centroid", "geometry": geom})

    if not data:
        print("No data to plot.")
        return

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # Convert to Web Mercator for Contextily
    try:
        gdf_web = gdf.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))

    # --- Plot 1: Raw Points (All points) ---
    print("Plotting raw points...")
    # Create GDF for all raw points
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in trajectory]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(ax=ax1, color='blue', markersize=5, alpha=0.5, label='GPS Fixes')
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Raw GPS Trajectory ({input_filename})")
        
        # Add legend
        # Create custom legend elements
        import matplotlib.lines as mlines
        blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=5, alpha=0.5, label='GPS Fixes')
        ax1.legend(handles=[blue_dot])
        
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Compressed (RED Moves as Lines, BLACK Centroids) ---
    print("Plotting compressed trajectory (Lines)...")
    # Plot Moves (Red Lines)
    moves = gdf_web[gdf_web["type"] == "Move"]
    if not moves.empty:
        moves.plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label='Move')

    # Plot Centroids (Black)
    centroids = gdf_web[gdf_web["type"] == "Centroid"]
    if not centroids.empty:
        centroids.plot(ax=ax2, color='black', markersize=100, marker='o', zorder=5, label='Centroid')

    # Add Basemap
    try:
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Basemap error plot 2: {e}")

    ax2.set_axis_off()
    ax2.set_title(f"Compressed (Lines) ({input_filename})")
    
    # Custom legend for plot 2
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Compressed Move')
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='Stop Centroid')
    ax2.legend(handles=[red_line, black_dot])


    # --- Plot 3: Compressed (RED Moves as Points, BLACK Centroids) ---
    print("Plotting compressed trajectory (Points)...")
    
    # Needs a new GDF where moves are just points, not LineStrings
    data_points = []
    for item in processed_items:
        if isinstance(item, Move):
            for p in item.points:
                geom = ShapelyPoint(p.lon, p.lat)
                data_points.append({"type": "MovePoint", "geometry": geom})
        elif isinstance(item, CompressedStop):
            geom = ShapelyPoint(item.centroid.lon, item.centroid.lat)
            data_points.append({"type": "Centroid", "geometry": geom})

    if data_points:
        gdf_points = gpd.GeoDataFrame(data_points, crs="EPSG:4326")
        try:
            gdf_points_web = gdf_points.to_crs(epsg=3857)
            
            # Plot Move Points
            move_points = gdf_points_web[gdf_points_web["type"] == "MovePoint"]
            if not move_points.empty:
                move_points.plot(ax=ax3, color='red', markersize=15, alpha=0.7, label='Move Point')
                
            # Plot Centroids
            stop_centroids = gdf_points_web[gdf_points_web["type"] == "Centroid"]
            if not stop_centroids.empty:
                stop_centroids.plot(ax=ax3, color='black', markersize=100, marker='o', zorder=5, label='Centroid')
            
            ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
             print(f"Error plotting plot 3: {e}")
    
    ax3.set_axis_off()
    ax3.set_title(f"Compressed (Points) ({input_filename})")

    # Custom legend for plot 3
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=5, label='Move Point')
    ax3.legend(handles=[red_dot, black_dot])

    plt.tight_layout()
    output_img = os.path.join(output_dir, f"full_flow_demo_{input_filename}.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    main()
