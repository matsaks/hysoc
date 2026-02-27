import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to sys.path to find benchmarks and src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.stream import TrajectoryStream
from hysoc.core.segment import Move
from benchmarks.oracles.stc import STCOracle

def main():
    # 1. Load Data
    data_path = os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv")
    
    print(f"Loading data from {data_path}...")
    # Use TrajectoryStream to easily load map-matched points with osm_way_id
    stream = TrajectoryStream(
        filepath=data_path,
        col_mapping={
            'lat': 'matched_latitude',
            'lon': 'matched_longitude',
            'timestamp': 'time',
            'obj_id': 'obj_id',
            'road_id': 'osm_way_id'
        }
    )
    trajectory = list(stream.stream())
    print(f"Loaded {len(trajectory)} points.")

    if not trajectory:
        print("No points loaded. Exiting.")
        return

    # 2. Compress using STC (Semantic Trajectory Compression)
    print("Running STCOracle...")
    # We treat the entire trajectory as one Move segment for this pure STC demonstration
    # In a full hybrid system, STSS would first split into Stops/Moves.
    move_segment = Move(points=trajectory)
    oracle = STCOracle()
    compressed_points = oracle.process(move_segment)
    
    print(f"Compression complete.")
    print(f" - Original points: {len(trajectory)}")
    print(f" - Compressed points (chunks): {len(compressed_points)}")
    print(f" - Compression Ratio: {len(trajectory) / len(compressed_points):.2f}x")

    # 4. Save to CSV
    script_name = "demo_07_offline_stc_oracle"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Raw CSV
    raw_csv = os.path.join(output_dir, "raw_trajectory.csv")
    print(f"Saving raw trajectory to {raw_csv}...")
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id"])
        for p in trajectory:
            writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id])    

    # Save Compressed CSV
    output_csv = os.path.join(output_dir, "compressed_trajectory.csv")
    print(f"Saving STC compressed trajectory to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time", "road_id", "type"])
        
        for p in compressed_points:
             writer.writerow([
                p.lat, p.lon,
                p.timestamp.strftime("%Y-%m-%d %H:%M:%S"), p.road_id, "CHUNK_START"
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

    # Create GDF for compressed points
    data = []
    # Create the logical compressed path by connecting the chunk start points
    if len(compressed_points) >= 2:
        coords = [(p.lon, p.lat) for p in compressed_points]
        geom = LineString(coords)
        data.append({"type": "CompressedPath", "geometry": geom})
        
    for p in compressed_points:
        geom = ShapelyPoint(p.lon, p.lat)
        data.append({"type": "ChunkNode", "geometry": geom})

    if not data:
        print("No data to plot.")
        return

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    try:
        gdf_web = gdf.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # --- Plot 1: Raw Points ---
    print("Plotting raw points...")
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in trajectory]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(ax=ax1, color='blue', markersize=5, alpha=0.5, label='Map-Matched GPS Fixes')
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Raw GPS Trajectory ({input_filename})")
        
        # Plot continuous line between raw points
        if len(trajectory) >= 2:
            raw_line = LineString([(p.lon, p.lat) for p in trajectory])
            gpd.GeoDataFrame([{"geometry": raw_line}], crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax1, color='blue', linewidth=1, alpha=0.3)
            
        ax1.legend()
    except Exception as e:
        print(f"Error plotting plot 1: {e}")

    # --- Plot 2: Compressed Chunks ---
    print("Plotting STC compressed trajectory...")
    path = gdf_web[gdf_web["type"] == "CompressedPath"]
    if not path.empty:
        path.plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label='Chunk Path')

    nodes = gdf_web[gdf_web["type"] == "ChunkNode"]
    if not nodes.empty:
        nodes.plot(ax=ax2, color='black', markersize=50, marker='o', zorder=5, label='Chunk Reference Point')

    try:
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Basemap error: {e}")

    ax2.set_axis_off()
    ax2.set_title(f"STC Representation ({input_filename}) - {len(compressed_points)} nodes")
    
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Logical Path')
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=8, label='Chunk Start')
    ax2.legend(handles=[red_line, black_dot])

    plt.tight_layout()
    output_img = os.path.join(output_dir, "plot.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    main()
