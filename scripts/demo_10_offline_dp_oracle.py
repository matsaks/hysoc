import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from benchmarks.oracles.dp import DPOracle
from hysoc.modules.stop_compression.compressor import StopCompressor, CompressedStop
from hysoc.metrics import calculate_compression_ratio, calculate_sed_stats

def load_trajectory(filepath: str) -> List[Point]:
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

def main():
    data_path = os.path.join(project_root, "data", "raw", "subset_50", "5605172.csv")
    print(f"Loading data from {data_path}...")
    trajectory = load_trajectory(data_path)
    print(f"Loaded {len(trajectory)} points.")

    if not trajectory:
        return

    # 1. Segment using STSS
    print("Running STSSOracleSklearn...")
    oracle = STSSOracleSklearn(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    segments = oracle.process(trajectory)
    print(f"Segmentation complete. Found {len(segments)} segments.")
    
    n_stops = sum(1 for s in segments if isinstance(s, Stop))
    n_moves = sum(1 for s in segments if isinstance(s, Move))
    print(f" - Stops: {n_stops}")
    print(f" - Moves: {n_moves}")

    # 2. Compress using DP Oracle
    print("Compressing using Douglas-Peucker Oracle...")
    stop_compressor = StopCompressor()
    dp_oracle = DPOracle(epsilon_meters=15.0)  # Moderate geometric tolerance
    
    processed_items = []

    for seg in segments:
        if isinstance(seg, Stop):
            compressed_stop = stop_compressor.compress(seg.points)
            processed_items.append(compressed_stop)
        elif isinstance(seg, Move):
            compressed_points = dp_oracle.process(seg)
            processed_items.append(Move(points=compressed_points))

    print(f"Compression complete. Result stream length: {len(processed_items)} items.")

    # 3. Calculate Metrics
    print("Calculating evaluation metrics...")
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
    
    print(f"Metrics Calculated:")
    print(f" - Compression Ratio: {compression_ratio:.2f}")
    print(f" - Stored Points: {stored_points_count} (Original: {len(trajectory)})")
    print(f" - Average SED: {sed_stats['average_sed']:.2f} m")

    # 4. Save Outputs
    script_name = "demo_10_offline_dp_oracle"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(project_root, "data", "processed", script_name, f"{timestamp}_{input_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "compressed_trajectory.csv")
    print(f"Saving compressed trajectory to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "start_time", "end_time", "type"])
        for item in processed_items:
            if isinstance(item, CompressedStop):
                writer.writerow([item.centroid.lat, item.centroid.lon, item.start_time, item.end_time, "STOP"])
            elif isinstance(item, Move):
                for p in item.points:
                     writer.writerow([p.lat, p.lon, p.timestamp, p.timestamp, "MOVE_POINT"])

    # 5. Visualize
    print("Generating visualization...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point as ShapelyPoint, LineString
        import contextily as ctx
    except ImportError:
        print("geopandas, shapely, or contextily not found.")
        return

    data = []
    for item in processed_items:
        if isinstance(item, Move):
            if not item.points:
                continue
            if len(item.points) < 2:
                geom = ShapelyPoint(item.points[0].lon, item.points[0].lat)
            else:
                geom = LineString([(p.lon, p.lat) for p in item.points])
            data.append({"type": "Move", "geometry": geom})
        elif isinstance(item, CompressedStop):
            geom = ShapelyPoint(item.centroid.lon, item.centroid.lat)
            data.append({"type": "Centroid", "geometry": geom})

    if not data:
        return

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    try:
        gdf_web = gdf.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error converting CRS: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Raw Points
    raw_points_data = [{'geometry': ShapelyPoint(p.lon, p.lat)} for p in trajectory]
    gdf_raw = gpd.GeoDataFrame(raw_points_data, crs="EPSG:4326")
    try:
        gdf_raw_web = gdf_raw.to_crs(epsg=3857)
        gdf_raw_web.plot(ax=ax1, color='blue', markersize=5, alpha=0.5, label='GPS Fixes')
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_axis_off()
        ax1.set_title(f"Raw GPS Trajectory ({input_filename})")
        
        import matplotlib.lines as mlines
        blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=5, alpha=0.5, label='GPS Fixes')
        ax1.legend(handles=[blue_dot])
    except Exception:
        pass

    # DP Compressed
    moves = gdf_web[gdf_web["type"] == "Move"]
    if not moves.empty:
        moves.plot(ax=ax2, color='red', linewidth=2, alpha=0.7, label='Move')

    centroids = gdf_web[gdf_web["type"] == "Centroid"]
    if not centroids.empty:
        centroids.plot(ax=ax2, color='black', markersize=100, marker='o', zorder=5, label='Centroid')

    try:
        ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    except Exception:
        pass

    ax2.set_axis_off()
    ax2.set_title(f"Douglas-Peucker Oracle ({input_filename})")
    
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Compressed Move')
    black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Stop Centroid')
    ax2.legend(handles=[red_line, black_dot])

    plt.tight_layout()
    output_img = os.path.join(output_dir, "plot.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")

    # --- Plot Metrics and Error Analysis (New Figure) ---
    print("Generating metrics visualization...")
    fig2 = plt.figure(figsize=(12, 10))
    
    # Layout: Top half for text, Bottom half for error plot
    gs = fig2.add_gridspec(2, 1, height_ratios=[1, 2])
    
    # Text Axis
    ax_text = fig2.add_subplot(gs[0])
    ax_text.axis('off')
    
    text_str = (
        f"Evaluation Metrics\n"
        f"------------------\n"
        f"Original Points: {len(trajectory)}\n"
        f"Stored Points: {stored_points_count}\n"
        f"Compression Ratio: {compression_ratio:.2f}:1\n\n"
        f"SED Statistics\n"
        f"--------------\n"
        f"Average Error: {sed_stats['average_sed']:.2f} m\n"
        f"Max Error: {sed_stats['max_sed']:.2f} m\n"
        f"RMSE: {sed_stats['rmse']:.2f} m"
    )
    
    ax_text.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=14, family='monospace')
    
    # Error Plot Axis
    ax_error = fig2.add_subplot(gs[1])
    
    errors = sed_stats.get('sed_errors', [])
    if errors:
        # Plot error vs index
        ax_error.plot(errors, color='purple', alpha=0.7, linewidth=1)
        ax_error.set_title("SED Error per Point")
        ax_error.set_xlabel("Original Point Index")
        ax_error.set_ylabel("Synchronized Euclidean Distance (m)")
        ax_error.grid(True, linestyle='--', alpha=0.5)
        
        # Add mean line
        ax_error.axhline(y=sed_stats['average_sed'], color='green', linestyle='--', label=f'Mean: {sed_stats["average_sed"]:.2f}m')
        ax_error.legend()
    else:
        ax_error.text(0.5, 0.5, "No error data available.", ha='center', va='center')
        
    plt.tight_layout()
    output_metrics_img = os.path.join(output_dir, "metrics.png")
    fig2.savefig(output_metrics_img, dpi=300, bbox_inches='tight')
    print(f"Metrics visualization saved to {output_metrics_img}")

if __name__ == "__main__":
    main()
