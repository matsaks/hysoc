import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import osmnx as ox

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from hysoc.core.stream import TrajectoryStream
from hysoc.core.segment import Stop, Move
from hysoc.modules.segmentation.step import STEPSegmenter
from hysoc.modules.stop_compression.compressor import StopCompressor
from hysoc.modules.move_compression.trace import TraceCompressor, TraceConfig
from hysoc.modules.map_matching.matcher import OnlineMapMatcher
from hysoc.modules.map_matching.wrapper import MapMatchedStreamWrapper

class HYSOCVisualizer:
    def __init__(self, filepath, max_eps=20.0, min_duration=30.0, interval=50, batch_size=1):
        self.filepath = filepath
        self.max_eps = max_eps
        self.min_duration = min_duration
        self.interval = interval
        self.points_batch_size = batch_size
        
        # Initialize components
        # 1. First Pass: Load all points to get bounding box and download map
        print("Pre-scanning file for map bounding box...")
        temp_stream = TrajectoryStream(filepath, default_obj_id='demo_obj')
        raw_points = list(temp_stream.stream()) # Exhaust iterator
        
        if not raw_points:
            raise ValueError(f"No points found in {filepath}")
            
        lats = [p.lat for p in raw_points]
        lons = [p.lon for p in raw_points]
        north, south = max(lats) + 0.005, min(lats) - 0.005
        east, west = max(lons) + 0.005, min(lons) - 0.005
        
        print(f"Downloading street graph for bbox: N:{north:.4f}, S:{south:.4f}, E:{east:.4f}, W:{west:.4f}...")
        self.G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
        print(f"Graph downloaded. Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
        
        # 2. Setup Map Matcher and Streaming Pipeline
        matcher = OnlineMapMatcher(
            G=self.G, 
            window_size=15, 
            max_dist=50, 
            max_dist_init=100, 
            min_prob_norm=0.001
        )
        
        # Re-initialize stream for actual processing
        self.streamor = TrajectoryStream(filepath, default_obj_id='demo_obj')
        # Wrap the raw stream with Map Matcher
        self.stream_iter = MapMatchedStreamWrapper(self.streamor.stream(), matcher).stream()
        
        self.segmenter = STEPSegmenter(max_eps=max_eps, min_duration_seconds=min_duration)
        self.stop_compressor = StopCompressor()
        self.move_compressor = TraceCompressor(TraceConfig(gamma=10.0))
        
        # Stats
        self.total_points = 0
        self.compressed_size_estimate = 0
        self.segments_count = {'Stop': 0, 'Move': 0}
        self.status_text = "Initializing..."
        
        # Data storage for plotting
        self.all_processed_points = []
        self.stops = [] # List of Stop objects
        self.moves = [] # List of Move objects
        self.current_buffer = [] # Points currently in buffer
        
        # Setup Plot
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[4, 1])
        
        # Map View (Top Left)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_map.set_title(f"HYSOC Simulation (max_eps={max_eps}m, min_dur={min_duration}s)")
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")

        # Plot Background Map (Road Network)
        print("Plotting road network background...")
        # Plot edges
        for u, v, data in self.G.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                self.ax_map.plot(xs, ys, color='silver', linewidth=1, alpha=0.5, zorder=1)
            else:
                # If no geometry, plot straight line between nodes
                x1 = self.G.nodes[u]['x']
                y1 = self.G.nodes[u]['y']
                x2 = self.G.nodes[v]['x']
                y2 = self.G.nodes[v]['y']
                self.ax_map.plot([x1, x2], [y1, y2], color='silver', linewidth=1, alpha=0.5, zorder=1)

        # Plot Elements
        self.line_raw, = self.ax_map.plot([], [], 'k.', markersize=2, alpha=0.3, label='Matched Points', zorder=2)
        self.line_buffer, = self.ax_map.plot([], [], 'y.', markersize=5, label='Active Buffer', zorder=3)
        self.line_moves, = self.ax_map.plot([], [], 'b-', linewidth=2.0, alpha=0.8, label='Move Segments', zorder=4)
        self.scatter_stops = self.ax_map.scatter([], [], c='r', s=60, marker='o', label='Stop Segments', zorder=5)
        self.ax_map.legend(loc='upper right')
        
        # Set aspect ratio to equal for map
        self.ax_map.set_aspect('equal')
        
        # Info Panel (Right Column)
        self.ax_info = self.fig.add_subplot(gs[:, 1])
        self.ax_info.axis('off')
        
        # Processing Log (Bottom Left)
        self.ax_log = self.fig.add_subplot(gs[1, 0])
        self.ax_log.axis('off')
        self.log_text_obj = self.ax_log.text(0.01, 0.9, "", va='top', fontsize=10, fontfamily='monospace')

        # self.points_batch_size is now passed in init
        self.finished = False

    def update(self, frame):
        if self.finished:
            return
        
        new_segments = []
        try:
            for _ in range(self.points_batch_size):
                point = next(self.stream_iter)
                self.total_points += 1
                self.all_processed_points.append(point)
                
                # Run HYSOC Algorithm Step
                produced = self.segmenter.process_point(point)
                new_segments.extend(produced)
                
        except StopIteration:
            if not self.finished: # Only run flush once
                produced = self.segmenter.flush()
                new_segments.extend(produced)
                self.finished = True
                self.status_text = "Stream Finished."
        
        # Handle New Segments
        log_messages = []
        for seg in new_segments:
            if isinstance(seg, Stop):
                self.segments_count['Stop'] += 1
                compressed = self.stop_compressor.compress(seg.points)
                self.stops.append(compressed)
                self.compressed_size_estimate += 1 # 1 point
                log_messages.append(f"STOP detected! ({len(seg.points)} pts) -> Centroid")
            elif isinstance(seg, Move):
                self.segments_count['Move'] += 1
                self.moves.append(seg)
                try:
                    compressed = self.move_compressor.compress(seg.points)
                    # Estimate compressed size
                    if compressed:
                        # compressed is a dict, let's just count top-level items as "units" or 10%
                        self.compressed_size_estimate += max(1, len(seg.points) // 10) 
                except Exception as e:
                    print(f"Move compression failed: {e}")
                    # Fallback estimate
                    self.compressed_size_estimate += len(seg.points)
                
                log_messages.append(f"MOVE Segment finalized ({len(seg.points)} pts)")

        # Update Buffer Vis
        # Step segmenter cache holds current working set
        cached_items = self.segmenter.cache
        buffer_points = [item[0] for item in cached_items]
        
        # Update Plots
        # 1. Raw Trail (Optimization: only plot last N points to avoid lag, or decimate)
        # We'll plot all points for now, but maybe decimate for speed if needed
        lats = [p.lat for p in self.all_processed_points]
        lons = [p.lon for p in self.all_processed_points]
        self.line_raw.set_data(lons, lats)
        
        # 2. Buffer
        if buffer_points:
            buf_lats = [p.lat for p in buffer_points]
            buf_lons = [p.lon for p in buffer_points]
            self.line_buffer.set_data(buf_lons, buf_lats)
        else:
            self.line_buffer.set_data([], [])

        # 3. Moves
        # To make it fast, we can concatenate moves or plot them as segments.
        # matplotlib line plot handles NaN breaks for disjoint lines
        move_lons = []
        move_lats = []
        for m in self.moves:
            for p in m.points:
                move_lons.append(p.lon)
                move_lats.append(p.lat)
            move_lons.append(float('nan'))
            move_lats.append(float('nan'))
        self.line_moves.set_data(move_lons, move_lats)
        
        # 4. Stops
        if self.stops:
            stop_lons = [s.centroid.lon for s in self.stops]
            stop_lats = [s.centroid.lat for s in self.stops]
            self.scatter_stops.set_offsets(list(zip(stop_lons, stop_lats)))

        # Update View Limits to follow the action
        if buffer_points:
            last_p = buffer_points[-1]
            # Simple follow cam
            margin = 0.01
            # self.ax_map.set_xlim(last_p.lon - margin, last_p.lon + margin)
            # self.ax_map.set_ylim(last_p.lat - margin, last_p.lat + margin)
            
            # Or fit all data
            if self.total_points > 0 and self.total_points % 50 == 0: # Update limits occasionally
                 self.ax_map.relim()
                 self.ax_map.autoscale_view()

        # Update Info Panel
        comp_ratio = self.total_points / max(1, self.compressed_size_estimate)
        
        info_text = (
            f"STATUS: {self.status_text}\n\n"
            f"PARAMETERS:\n"
            f"  Max Eps (Dist): {self.max_eps} m\n"
            f"  Min Duration:   {self.min_duration} s\n"
            f"  Move Gamma:     10.0\n\n"
            f"STATISTICS:\n"
            f"  Total Points:   {self.total_points}\n"
            f"  Stops Found:    {self.segments_count['Stop']}\n"
            f"  Moves Found:    {self.segments_count['Move']}\n"
            f"  Est. Comp Size: {self.compressed_size_estimate}\n"
            f"  COMPRESSION RATIO: {comp_ratio:.2f}x\n"
        )
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, 
                          va='top', fontsize=12, family='monospace')
        
        # Update Log
        if log_messages:
            self.log_text_obj.set_text("\n".join(log_messages[-3:])) # Show last 3 messages

        return self.line_raw, self.line_buffer, self.line_moves, self.scatter_stops

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=self.interval, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYSOC Algorithm Visualization")
    parser.add_argument("--file", type=str, help="Path to trajectory CSV file")
    
    # Defaults are None so that class defaults are used if not provided
    parser.add_argument("--eps", type=float, default=None, help="Max distance threshold (meters)")
    parser.add_argument("--min_dur", type=float, default=None, help="Min duration threshold (seconds)")
    parser.add_argument("--interval", type=int, default=None, help="Interval between frames (ms)")
    parser.add_argument("--batch", type=int, default=None, help="Points processed per frame")
    
    args = parser.parse_args()
    
    # Default file logic
    target_file = args.file
    if not target_file:
        # Try to find a default file
        default_path = os.path.join(project_root, "data/raw/subset_50/4494499.csv")
        if os.path.exists(default_path):
            target_file = default_path
            print(f"No file specified. Using default: {target_file}")
        else:
            print("Error: No file specified and default file not found.")
            sys.exit(1)

    viz_kwargs = {}
    if args.eps is not None:
        viz_kwargs['max_eps'] = args.eps
    if args.min_dur is not None:
        viz_kwargs['min_duration'] = args.min_dur
    if args.interval is not None:
        viz_kwargs['interval'] = args.interval
    if args.batch is not None:
        viz_kwargs['batch_size'] = args.batch

    viz = HYSOCVisualizer(target_file, **viz_kwargs)
    viz.run()
