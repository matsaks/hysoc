import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from hysoc.modules.segmentation.step import STEPSegmenter
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from hysoc.modules.stop_compression.compressor import StopCompressor, CompressedStop
from hysoc.modules.move_compression.squish import SquishCompressor
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

def compress_segment(seg: Segment, stop_compressor: StopCompressor, move_compressor: SquishCompressor):
    if isinstance(seg, Stop):
        return stop_compressor.compress(seg.points)
    elif isinstance(seg, Move):
        dynamic_capacity = max(10, int(len(seg.points) * 0.3))
        compressed_points = move_compressor.compress(seg.points, capacity=dynamic_capacity)
        return Move(points=compressed_points)
    return None

def calculate_all_metrics(trajectory, processed_items):
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
    
    return {
        'compression_ratio': compression_ratio,
        'stored_points': stored_points_count,
        'original_points': len(trajectory),
        'sed_stats': sed_stats
    }

def print_metrics(name, m):
    print(f"\n--- Metrics for {name} ---")
    print(f" Compression Ratio: {m['compression_ratio']:.2f}")
    print(f" Stored Points: {m['stored_points']} (Original: {m['original_points']})")
    print(f" Average SED: {m['sed_stats']['average_sed']:.2f} m")
    print(f" Max SED: {m['sed_stats']['max_sed']:.2f} m")
    print(f" RMSE: {m['sed_stats']['rmse']:.2f} m")

def get_plot_data(processed_items):
    try:
        from shapely.geometry import Point as ShapelyPoint, LineString
        import geopandas as gpd
    except ImportError:
        return None
        
    data_lines = []
    data_points = []
    
    for item in processed_items:
        if isinstance(item, Move):
            if not item.points:
                continue
            # For lines
            if len(item.points) < 2:
                geom_l = ShapelyPoint(item.points[0].lon, item.points[0].lat)
            else:
                coords = [(p.lon, p.lat) for p in item.points]
                geom_l = LineString(coords)
            data_lines.append({"type": "Move", "geometry": geom_l})
            
            # For points
            for p in item.points:
                geom_p = ShapelyPoint(p.lon, p.lat)
                data_points.append({"type": "MovePoint", "geometry": geom_p})
                
        elif isinstance(item, CompressedStop):
            geom_c = ShapelyPoint(item.centroid.lon, item.centroid.lat)
            data_lines.append({"type": "Centroid", "geometry": geom_c})
            data_points.append({"type": "Centroid", "geometry": geom_c})

    if not data_lines:
        return None
        
    gdf_lines = gpd.GeoDataFrame(data_lines, crs="EPSG:4326")
    gdf_points = gpd.GeoDataFrame(data_points, crs="EPSG:4326")
    
    try:
        gdf_lines_web = gdf_lines.to_crs(epsg=3857)
        gdf_points_web = gdf_points.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return None
        
    return gdf_lines_web, gdf_points_web

def plot_ax(ax, gdf_web, is_points=False, title=""):
    import contextily as ctx
    if is_points:
        move_points = gdf_web[gdf_web["type"] == "MovePoint"]
        if not move_points.empty:
            move_points.plot(ax=ax, color='red', markersize=15, alpha=0.7, label='Move Point')
    else:
        moves = gdf_web[gdf_web["type"] == "Move"]
        if not moves.empty:
            moves.plot(ax=ax, color='red', linewidth=2, alpha=0.7, label='Move')

    centroids = gdf_web[gdf_web["type"] == "Centroid"]
    if not centroids.empty:
        centroids.plot(ax=ax, color='black', markersize=100, marker='o', zorder=5, label='Centroid')

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        pass

    ax.set_axis_off()
    ax.set_title(title)
        
    if is_points:
        red_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Move Point')
    else:
        red_legend = mlines.Line2D([], [], color='red', linewidth=2, label='Compressed Move')
        
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Stop Centroid')
    ax.legend(handles=[red_legend, black_dot])


def main():
    data_path = os.path.join(project_root, "data", "raw", "subset_50", "4494499.csv")
    print(f"Loading data from {data_path}...")
    trajectory = load_trajectory(data_path)
    print(f"Loaded {len(trajectory)} points.")

    if not trajectory:
        return

    # 1. OFFLINE (STSS)
    print("\n[1] Running STSSOracleSklearn (Offline)...")
    stss_oracle = STSSOracleSklearn(min_samples=5, max_eps=10, min_duration_seconds=10)
    stss_segments = stss_oracle.process(trajectory)
    
    stop_compressor = StopCompressor()
    move_compressor_stss = SquishCompressor(capacity=50)
    stss_processed = []
    for seg in stss_segments:
        comp = compress_segment(seg, stop_compressor, move_compressor_stss)
        if comp: stss_processed.append(comp)

    # 2. ONLINE (STEP)
    print("\n[2] Running STEPSegmenter (Online Streaming)...")
    step_segmenter = STEPSegmenter(max_eps=10, min_duration_seconds=10, grid_size_meters=5.0)
    move_compressor_step = SquishCompressor(capacity=50) # Use independent compressor
    step_processed = []
    
    for point in trajectory:
        for seg in step_segmenter.process_point(point):
            comp = compress_segment(seg, stop_compressor, move_compressor_step)
            if comp: step_processed.append(comp)
            
    for seg in step_segmenter.flush():
        comp = compress_segment(seg, stop_compressor, move_compressor_step)
        if comp: step_processed.append(comp)

    # 3. METRICS
    stss_metrics = calculate_all_metrics(trajectory, stss_processed)
    step_metrics = calculate_all_metrics(trajectory, step_processed)
    
    print_metrics("Offline STSS", stss_metrics)
    print_metrics("Online STEP", step_metrics)

    # 4. VISUALIZATION
    print("\nGenerating comparative visualization...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found.")
        return

    stss_data = get_plot_data(stss_processed)
    step_data = get_plot_data(step_processed)
    
    if not stss_data or not step_data:
        print("Failed to prepare plot data.")
        return

    stss_lines_web, stss_points_web = stss_data
    step_lines_web, step_points_web = step_data

    # Create a 2x3 Figure: 
    # Row 1: STSS (Raw, Lines, Points)
    # Row 2: STEP (Text Metrics, Lines, Points) -> let's just make Raw identically in col 1 or put metrics in col 1.
    fig, axes = plt.subplots(2, 3, figsize=(36, 24))

    # --- Plot Raw Data (Col 1) ---
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in trajectory]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        for i in range(2):
            gdf_raw_web.plot(ax=axes[i, 0], color='blue', markersize=5, alpha=0.5, label='GPS Fixes')
            ctx.add_basemap(axes[i, 0], source=ctx.providers.CartoDB.Positron)
            axes[i, 0].set_axis_off()
            blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=5, alpha=0.5, label='GPS Fixes')
            axes[i, 0].legend(handles=[blue_dot])
            
        axes[0, 0].set_title("Raw Trajectory (For STSS)")
        axes[1, 0].set_title("Raw Trajectory (For STEP)")
    except Exception as e:
        print(f"Error plotting Raw: {e}")

    # --- Plot STSS (Row 0, Cols 1, 2) ---
    plot_ax(axes[0, 1], stss_lines_web, is_points=False, title="Offline STSS - Compressed Lines")
    plot_ax(axes[0, 2], stss_points_web, is_points=True, title="Offline STSS - Compressed Points")

    # --- Plot STEP (Row 1, Cols 1, 2) ---
    plot_ax(axes[1, 1], step_lines_web, is_points=False, title="Online STEP - Compressed Lines")
    plot_ax(axes[1, 2], step_points_web, is_points=True, title="Online STEP - Compressed Points")

    plt.tight_layout()
    
    script_name = "demo_04_compare_stss_vs_step"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_img = os.path.join(output_dir, "plot.png")
    
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Comparative visualization saved to {output_img}")
    
    # Save Raw Trajectory CSV
    raw_csv = os.path.join(output_dir, "raw_trajectory.csv")
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "time"])
        for p in trajectory:
            writer.writerow([p.lat, p.lon, p.timestamp.strftime("%Y-%m-%d %H:%M:%S")])
            
    # Save Compressed Trajectories CSVs helper
    def save_compressed(items, filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "start_time", "end_time", "type"])
            for item in items:
                if isinstance(item, CompressedStop):
                    writer.writerow([item.centroid.lat, item.centroid.lon, item.start_time, item.end_time, "STOP"])
                elif isinstance(item, Move):
                    for p in item.points:
                        writer.writerow([p.lat, p.lon, p.timestamp, p.timestamp, "MOVE_POINT"])
                        
    save_compressed(stss_processed, os.path.join(output_dir, "stss_compressed.csv"))
    save_compressed(step_processed, os.path.join(output_dir, "step_compressed.csv"))
    print("Saved raw and compressed CSV trajectories.")
    
    # Save a separate text file with metrics next to it for easy reading
    metrics_txt = os.path.join(output_dir, "metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"--- Metrics for Offline STSS ---\n")
        f.write(f" Compression Ratio: {stss_metrics['compression_ratio']:.2f}\n")
        f.write(f" Stored Points: {stss_metrics['stored_points']} (Original: {stss_metrics['original_points']})\n")
        f.write(f" Average SED: {stss_metrics['sed_stats']['average_sed']:.2f} m\n")
        f.write(f" Max SED: {stss_metrics['sed_stats']['max_sed']:.2f} m\n")
        f.write(f" RMSE: {stss_metrics['sed_stats']['rmse']:.2f} m\n\n")
        
        f.write(f"--- Metrics for Online STEP ---\n")
        f.write(f" Compression Ratio: {step_metrics['compression_ratio']:.2f}\n")
        f.write(f" Stored Points: {step_metrics['stored_points']} (Original: {step_metrics['original_points']})\n")
        f.write(f" Average SED: {step_metrics['sed_stats']['average_sed']:.2f} m\n")
        f.write(f" Max SED: {step_metrics['sed_stats']['max_sed']:.2f} m\n")
        f.write(f" RMSE: {step_metrics['sed_stats']['rmse']:.2f} m\n")
        
    print(f"Comparative metrics text saved to {metrics_txt}")

if __name__ == "__main__":
    main()
