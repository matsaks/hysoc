
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
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

    # 2. Segment (Status: Segments with full point lists)
    print("Running STSSOracleSklearn...")
    # Parameters from notebook: min_samples=5, max_eps=10m, min_duration=10s
    oracle = STSSOracleSklearn(min_samples=5, max_eps=10, min_duration_seconds=10)
    segments = oracle.process(trajectory)
    print(f"Segmentation complete. Found {len(segments)} segments.")
    
    n_stops = sum(1 for s in segments if isinstance(s, Stop))
    n_moves = sum(1 for s in segments if isinstance(s, Move))
    print(f" - Stops: {n_stops}")
    print(f" - Moves: {n_moves}")

    # 3. Compress (Status: Moves + CompressedStops)
    print("Compressing stops...")
    compressor = StopCompressor()
    compressed_output = []
    
    # Store plain list for easy iteration later
    processed_items = []

    for seg in segments:
        if isinstance(seg, Stop):
            compressed_stop = compressor.compress(seg.points)
            processed_items.append(compressed_stop)
        elif isinstance(seg, Move):
            processed_items.append(seg)

    print(f"Compression complete. Result stream length: {len(processed_items)} items.")

    # 4. Save to CSV
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_csv = os.path.join(output_dir, f"compressed_output_{input_filename}.csv")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

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
        ax1.legend()
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Compressed (RED Moves, BLACK Centroids) ---
    print("Plotting compressed trajectory...")
    # Plot Moves (Red)
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
        print(f"Basemap error: {e}")

    ax2.set_axis_off()
    ax2.set_title(f"Compressed Trajectory ({input_filename})")
    
    # Custom legend for plot 2
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Move')
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='Centroid')
    ax2.legend(handles=[red_line, black_dot])

    plt.tight_layout()
    output_img = os.path.join(output_dir, f"stop_compression_demo_{input_filename}.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    main()
