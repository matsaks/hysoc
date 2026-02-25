import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from hysoc.modules.segmentation.step import STEPSegmenter
from hysoc.modules.stop_compression.compressor import StopCompressor, CompressedStop
from hysoc.metrics import calculate_compression_ratio, calculate_sed_stats

def load_trajectory(filepath: str) -> list[Point]:
    points = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dt = datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                points.append(Point(lat=lat, lon=lon, timestamp=dt, obj_id="demo_obj"))
            except ValueError as e:
                continue
    return points

def compress_segment(seg: Segment, stop_compressor: StopCompressor):
    if isinstance(seg, Stop):
        return stop_compressor.compress(seg.points)
    elif isinstance(seg, Move):
        return seg
    return None

def main():
    data_path = os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv")
    print(f"Loading data from {data_path}...")
    trajectory = load_trajectory(data_path)
    print(f"Loaded {len(trajectory)} points.")

    if not trajectory:
        return

    # Initialize streaming modules
    # Providing grid_size_meters explicitly for demonstration!
    segmenter = STEPSegmenter(max_eps=10, min_duration_seconds=10, grid_size_meters=5.0)
    stop_compressor = StopCompressor()

    processed_items = []
    print("Streaming trajectory points...")

    for point in trajectory:
        segments = segmenter.process_point(point)
        for seg in segments:
            compressed = compress_segment(seg, stop_compressor)
            if compressed is not None:
                processed_items.append(compressed)
                
    # End of stream flush
    flush_segments = segmenter.flush()
    for seg in flush_segments:
        compressed = compress_segment(seg, stop_compressor)
        if compressed is not None:
            processed_items.append(compressed)

    print(f"Streaming complete. Processed into {len(processed_items)} segments.")

    n_stops = sum(1 for s in processed_items if isinstance(s, CompressedStop))
    n_moves = sum(1 for s in processed_items if isinstance(s, Move))
    print(f" - Compressed Stops: {n_stops}")
    print(f" - Compressed Moves: {n_moves}")

    # --- Calculate Metrics (Similar to original demo script) ---
    print("Calculating metrics...")
    compressed_trajectory_for_sed = []
    stored_points_count = 0

    for item in processed_items:
        if isinstance(item, CompressedStop):
            p_start = Point(lat=item.centroid.lat, lon=item.centroid.lon, timestamp=item.start_time, obj_id=item.centroid.obj_id)
            p_end = Point(lat=item.centroid.lat, lon=item.centroid.lon, timestamp=item.end_time, obj_id=item.centroid.obj_id)
            compressed_trajectory_for_sed.extend([p_start, p_end])
            stored_points_count += 1
        elif isinstance(item, Move):
            if not item.points:
                continue
            compressed_trajectory_for_sed.extend(item.points)
            stored_points_count += len(item.points)
            
    compression_ratio = len(trajectory) / max(1, stored_points_count)
    sed_stats = calculate_sed_stats(trajectory, compressed_trajectory_for_sed)
    
    print("Metrics:")
    print(f" - Compression Ratio: {compression_ratio:.2f}")
    print(f" - Stored Points: {stored_points_count} (Original: {len(trajectory)})")
    print(f" - Average SED: {sed_stats['average_sed']:.2f} m")

    # 5. Visualize and Save
    print("Saving outputs and generating visualization...")
    script_name = "demo_03_online_step_stop_compression"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Raw Trajectory CSV
    raw_csv = os.path.join(output_dir, "raw_trajectory.csv")
    print(f"Saving raw trajectory to {raw_csv}...")
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time"])
        for p in trajectory:
            writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            
    # Save Compressed CSV
    output_csv = os.path.join(output_dir, "compressed_trajectory.csv")
    print(f"Saving compressed trajectory to {output_csv}...")
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
                    
    # Save Metrics Text
    metrics_txt = os.path.join(output_dir, "metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"--- Metrics for Online STEP ---\n")
        f.write(f" Compression Ratio: {compression_ratio:.2f}\n")
        f.write(f" Stored Points: {stored_points_count} (Original: {len(trajectory)})\n")
        f.write(f" Average SED: {sed_stats['average_sed']:.2f} m\n")
        f.write(f" Max SED: {sed_stats['max_sed']:.2f} m\n")
        f.write(f" RMSE: {sed_stats['rmse']:.2f} m\n")
    print(f"Metrics saved to {metrics_txt}")
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
    try:
        gdf_web = gdf.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))

    # --- Plot 1: Raw Points (All points) ---
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in trajectory]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(ax=ax1, color='blue', markersize=5, alpha=0.5, label='GPS Fixes')
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Raw GPS Trajectory ({input_filename})")
        import matplotlib.lines as mlines
        blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                markersize=5, alpha=0.5, label='GPS Fixes')
        ax1.legend(handles=[blue_dot])
    except Exception as e:
        print(f"Error plotting plot 1: {e}")
        
    # --- Plot 2: Compressed (RED Moves as Lines, BLACK Centroids) ---
    moves = gdf_web[gdf_web["type"] == "Move"]
    if not moves.empty:
        moves.plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label='Move')

    centroids = gdf_web[gdf_web["type"] == "Centroid"]
    if not centroids.empty:
        centroids.plot(ax=ax2, color='black', markersize=100, marker='o', zorder=5, label='Centroid')

    try:
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Basemap error plot 2: {e}")

    ax2.set_axis_off()
    ax2.set_title(f"Compressed (Lines) ({input_filename})")
    
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Compressed Move')
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='Stop Centroid')
    ax2.legend(handles=[red_line, black_dot])

    # --- Plot 3: Compressed (RED Moves as Points, BLACK Centroids) ---
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
            move_points = gdf_points_web[gdf_points_web["type"] == "MovePoint"]
            if not move_points.empty:
                move_points.plot(ax=ax3, color='red', markersize=15, alpha=0.7, label='Move Point')
                
            stop_centroids = gdf_points_web[gdf_points_web["type"] == "Centroid"]
            if not stop_centroids.empty:
                stop_centroids.plot(ax=ax3, color='black', markersize=100, marker='o', zorder=5, label='Centroid')
            
            ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
             print(f"Error plotting plot 3: {e}")
    
    ax3.set_axis_off()
    ax3.set_title(f"Compressed (Points) ({input_filename})")

    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=5, label='Move Point')
    ax3.legend(handles=[red_dot, black_dot])

    plt.tight_layout()
    output_img = os.path.join(output_dir, "plot.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

    print("Done!")

if __name__ == "__main__":
    main()
