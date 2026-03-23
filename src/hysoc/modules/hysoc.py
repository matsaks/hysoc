"""
HYSOC: Hybrid Trajectory Compression Framework

This module orchestrates the unified pipeline combining:
- Module I: Streaming Segmentation (STEP)
- Module II: Stop Compression (StopCompressor)
- Module III-A: Move Compression - Geometric (SquishCompressor)
- Module III-B: Move Compression - Network-Semantic (TraceCompressor)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Union, Tuple, Any
from enum import Enum
from datetime import datetime
import math

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from hysoc.modules.segmentation.step import STEPSegmenter
from hysoc.modules.stop_compression.compressor import StopCompressor, CompressedStop
from hysoc.modules.move_compression.squish import SquishCompressor
from hysoc.modules.move_compression.trace import TraceCompressor, TraceConfig
from hysoc.modules.map_matching.wrapper import MapMatchedStreamWrapper
from hysoc.modules.map_matching.matcher import OnlineMapMatcher
from hysoc.constants.geo_defaults import EARTH_RADIUS_M


class CompressionStrategy(Enum):
    """Enum for move compression strategies."""
    GEOMETRIC = "geometric"  # HYSOC-G: SquishCompressor
    NETWORK_SEMANTIC = "network_semantic"  # HYSOC-N: TraceCompressor


@dataclass
class HYSOCConfig:
    """
    Configuration for the unified HYSOC pipeline.
    
    Attributes:
        segmentation_strategy: Strategy for trajectory segmentation (currently only "step").
        move_compression_strategy: Strategy for move compression (GEOMETRIC or NETWORK_SEMANTIC).
        
        # Segmentation parameters
        stop_max_eps_meters: Distance threshold for stop detection.
        stop_min_duration_seconds: Minimum duration for a stop.
        
        # Stop compression parameters (always enabled)
        compress_stops: Whether to compress stop segments (always True for HYSOC).
        
        # Move compression parameters - GEOMETRIC (HYSOC-G)
        squish_capacity: Buffer capacity for SquishCompressor.
        
        # Move compression parameters - NETWORK_SEMANTIC (HYSOC-N)
        trace_config: Configuration object for TraceCompressor.
        osm_graph: OSMnx graph for map-matched trajectory compression (required for NETWORK_SEMANTIC).
        enable_map_matching: Whether to apply online map matching before TRACE compression.
    """
    
    # Strategy selection
    move_compression_strategy: CompressionStrategy = CompressionStrategy.GEOMETRIC
    
    # Segmentation parameters
    stop_max_eps_meters: float = 100.0  # meters
    stop_min_duration_seconds: float = 30.0  # 30 seconds
    
    # Stop compression (always enabled)
    compress_stops: bool = True
    
    # Move compression - GEOMETRIC (HYSOC-G)
    squish_capacity: int = 100
    
    # Move compression - NETWORK_SEMANTIC (HYSOC-N)
    trace_config: TraceConfig = field(default_factory=TraceConfig)
    osm_graph: Optional[Any] = None  # networkx.MultiDiGraph, optional for NETWORK_SEMANTIC
    enable_map_matching: bool = False


@dataclass
class CompressedSegment:
    """Represents a compressed segment (Stop or Move)."""
    segment_type: str  # "stop" or "move"
    original_segment: Union[Stop, Move]
    compressed_data: Any  # CompressedStop, List[Point], or TraceCompressor output
    compression_ratio: Optional[float] = None


@dataclass
class CompressedTrajectory:
    """
    Result of compressing a complete trajectory.
    
    Attributes:
        original_points: Original trajectory points.
        compressed_segments: List of CompressedSegment objects.
        total_original_points: Total points in original trajectory.
        total_compressed_points: Total points in compressed representation.
        overall_compression_ratio: (original - compressed) / original.
        compression_strategy: Which compression strategy was used.
        timestamp: When the compression occurred.
    """
    original_points: List[Point]
    compressed_segments: List[CompressedSegment]
    total_original_points: int
    total_compressed_points: int
    overall_compression_ratio: float
    compression_strategy: CompressionStrategy
    timestamp: datetime = field(default_factory=datetime.now)


class HYSOCCompressor:
    """
    Unified HYSOC trajectory compression orchestrator.
    
    This class coordinates all three modules to produce a compressed trajectory.
    It can operate in two modes:
    
    1. Streaming mode: process_point() for online compression
    2. Batch mode: compress() for offline compression
    """

    def __init__(self, config: HYSOCConfig = HYSOCConfig()):
        """
        Initialize the HYSOC compressor.
        
        Args:
            config: HYSOCConfig object specifying compression strategy and parameters.
            
        Raises:
            ValueError: If configuration is invalid (e.g., NETWORK_SEMANTIC without OSM graph).
        """
        self.config = config
        
        # Validate configuration
        if config.move_compression_strategy == CompressionStrategy.NETWORK_SEMANTIC:
            if config.enable_map_matching and config.osm_graph is None:
                raise ValueError(
                    "NETWORK_SEMANTIC strategy with map matching requires osm_graph. "
                    "Provide osm_graph via config.osm_graph."
                )
        
        # Initialize Module I: Segmentation
        self.segmenter = STEPSegmenter(
            max_eps=config.stop_max_eps_meters,
            min_duration_seconds=config.stop_min_duration_seconds,
        )
        
        # Initialize Module II: Stop Compression
        self.stop_compressor = StopCompressor()
        
        # Initialize Module III: Move Compression
        if config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
            self.move_compressor = SquishCompressor(capacity=config.squish_capacity)
        else:  # NETWORK_SEMANTIC
            self.move_compressor = TraceCompressor(config=config.trace_config)
        
        # Initialize optional map matcher for NETWORK_SEMANTIC
        self.map_matcher: Optional[OnlineMapMatcher] = None
        if config.enable_map_matching and config.osm_graph is not None:
            self.map_matcher = OnlineMapMatcher(config.osm_graph)
        
        # Storage for streaming mode
        self.all_segments: List[Segment] = []
        self.all_original_points: List[Point] = []

    def process_point(self, point: Point) -> Optional[List[CompressedSegment]]:
        """
        Streaming interface: process a single point and emit compressed segments if available.
        
        This is the online/incremental compression mode. Finished segments are compressed
        and emitted as they complete. Incomplete segments are buffered internally.
        
        Args:
            point: The incoming GPS point.
            
        Returns:
            List of CompressedSegment objects if segments finish, None otherwise.
            May return an empty list if no segments finished yet.
        """
        self.all_original_points.append(point)
        
        # Apply map matching if enabled
        if self.map_matcher is not None:
            matched_point = self.map_matcher.process_point(point)
            if matched_point is None:
                return None  # Still in map-matcher buffer
            point = matched_point
        
        # Segment the point
        segments = self.segmenter.process_point(point)
        
        # Compress finished segments
        compressed = []
        for seg in segments:
            compressed_seg = self._compress_segment(seg)
            if compressed_seg is not None:
                compressed.append(compressed_seg)
                self.all_segments.append(seg)
        
        return compressed if compressed else None

    def flush(self) -> CompressedTrajectory:
        """
        Terminate streaming and return the final compressed trajectory.
        
        Flushes remaining segments from both the segmenter and map matcher,
        then returns the complete compressed trajectory.
        
        Returns:
            CompressedTrajectory object with all compressed segments.
        """
        # Flush remaining segments from map matcher if active
        if self.map_matcher is not None:
            for point in self.map_matcher.flush():
                seg = self.segmenter.process_point(point)
                self.all_segments.extend(seg)
        
        # Flush remaining segments from segmenter
        final_segments = self.segmenter.flush()
        for seg in final_segments:
            self.all_segments.append(seg)
        
        # Compress all remaining segments
        compressed_segments = []
        for seg in self.all_segments:
            compressed_seg = self._compress_segment(seg)
            if compressed_seg is not None:
                compressed_segments.append(compressed_seg)
        
        # Calculate overall compression ratio
        total_original = len(self.all_original_points)
        total_compressed = self._count_compressed_points(compressed_segments)
        
        if total_original > 0:
            compression_ratio = (total_original - total_compressed) / total_original
        else:
            compression_ratio = 0.0
        
        return CompressedTrajectory(
            original_points=self.all_original_points,
            compressed_segments=compressed_segments,
            total_original_points=total_original,
            total_compressed_points=total_compressed,
            overall_compression_ratio=compression_ratio,
            compression_strategy=self.config.move_compression_strategy,
        )

    def _extract_retained_points_from_trace(self, points: List[Point]) -> List[Point]:
        """
        Extract retained points from a move segment using TRACE compression logic.
        
        Applies speed-based representation filtering to identify key points where
        speed changes significantly exceed the gamma threshold.
        
        Args:
            points: List of Point objects (map-matched move segment).
            
        Returns:
            List of retained Point objects identified by TRACE compression logic.
        """
        if not points or len(points) < 2:
            return points
        
        retained_points = []
        gamma = self.config.trace_config.gamma
        
        def lat_lon_dist(p1: Point, p2: Point) -> float:
            """Calculate distance between two points in meters."""
            lat1 = math.radians(p1.lat)
            lat2 = math.radians(p2.lat)
            dlat = lat2 - lat1
            dlon = math.radians(p2.lon - p1.lon)
            x = dlon * math.cos((lat1 + lat2) / 2.0)
            y = dlat
            return EARTH_RADIUS_M * math.sqrt(x * x + y * y)
        
        current_road_id = None
        last_stored_speed = -1.0
        
        for i, p in enumerate(points):
            # 1. Road Segment Change / Start
            if i == 0 or p.road_id != current_road_id:
                current_road_id = p.road_id
                retained_points.append(p)
                last_stored_speed = 0.0
                continue
            
            # 2. Same road segment - check for speed change
            prev_p = points[i - 1]
            dist = lat_lon_dist(prev_p, p)
            time_diff = (p.timestamp - prev_p.timestamp).total_seconds()
            
            if time_diff > 0:
                current_speed = dist / time_diff
            else:
                current_speed = last_stored_speed
            
            # Keep point if speed changed significantly
            if abs(current_speed - last_stored_speed) > gamma:
                retained_points.append(p)
                last_stored_speed = current_speed
        
        # Always ensure the last point is included
        if retained_points and retained_points[-1] != points[-1]:
            retained_points.append(points[-1])
        
        return retained_points

    def compress(self, points: List[Point]) -> CompressedTrajectory:
        """
        Batch/offline interface: compress a complete list of points.
        
        Args:
            points: List of Point objects representing a complete trajectory.
            
        Returns:
            CompressedTrajectory object with all compressed segments.
        """
        # Reset state for fresh compression
        self.segmenter = STEPSegmenter(
            max_eps=self.config.stop_max_eps_meters,
            min_duration_seconds=self.config.stop_min_duration_seconds,
        )
        self.all_segments = []
        self.all_original_points = []
        
        # Apply map matching if enabled
        if self.map_matcher is not None:
            points = self._apply_map_matching(points)
        
        # Process all points through segmenter
        for point in points:
            segments = self.segmenter.process_point(point)
            self.all_segments.extend(segments)
        
        # Flush any remaining segments
        final_segments = self.segmenter.flush()
        self.all_segments.extend(final_segments)
        
        # Compress all segments
        compressed_segments = []
        for seg in self.all_segments:
            compressed_seg = self._compress_segment(seg)
            if compressed_seg is not None:
                compressed_segments.append(compressed_seg)
        
        # Calculate overall compression ratio
        total_original = len(points)
        total_compressed = self._count_compressed_points(compressed_segments)
        
        if total_original > 0:
            compression_ratio = (total_original - total_compressed) / total_original
        else:
            compression_ratio = 0.0
        
        return CompressedTrajectory(
            original_points=self.all_original_points,
            compressed_segments=compressed_segments,
            total_original_points=total_original,
            total_compressed_points=total_compressed,
            overall_compression_ratio=compression_ratio,
            compression_strategy=self.config.move_compression_strategy,
        )

    def _compress_segment(self, seg: Segment) -> Optional[CompressedSegment]:
        """
        Route a segment to the appropriate compression module.
        
        Args:
            seg: Segment object (Stop or Move).
            
        Returns:
            CompressedSegment with compression results, or None if error.
        """
        if isinstance(seg, Stop):
            # Module II: Compress stop if enabled
            if self.config.compress_stops:
                compressed_data = self.stop_compressor.compress(seg.points)
                total_compressed = 1  # 1 point (centroid)
            else:
                compressed_data = seg
                total_compressed = len(seg.points)
            
            compression_ratio = (len(seg.points) - total_compressed) / len(seg.points) \
                if len(seg.points) > 0 else 0.0
            
            return CompressedSegment(
                segment_type="stop",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
            )
        
        elif isinstance(seg, Move):
            # Module III: Route move to appropriate compressor
            if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
                # HYSOC-G: Returns List[Point]
                compressed_data = self.move_compressor.compress(seg.points)
                total_compressed = len(compressed_data)
            else:  # NETWORK_SEMANTIC
                # HYSOC-N: TraceCompressor.compress() returns dict with E and V representations
                # Apply TRACE compression logic and extract retained points
                trace_result = self.move_compressor.compress(seg.points)
                
                # Extract retained points using speed-based representation logic
                retained_points = self._extract_retained_points_from_trace(seg.points)
                
                # Store both the retained points and the TRACE representation
                compressed_data = {
                    'trace_result': trace_result,  # E and V sequences
                    'retained_points': retained_points  # Geographic points for visualization
                }
                total_compressed = len(retained_points)
            
            compression_ratio = (len(seg.points) - total_compressed) / len(seg.points) \
                if len(seg.points) > 0 else 0.0
            
            return CompressedSegment(
                segment_type="move",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
            )
        
        return None

    def _apply_map_matching(self, points: List[Point]) -> List[Point]:
        """
        Apply map matching to a list of points.
        
        Args:
            points: List of Point objects.
            
        Returns:
            List of map-matched Point objects with road_id set.
        """
        if self.map_matcher is None:
            return points
        
        matched_points = []
        for point in points:
            matched = self.map_matcher.process_point(point)
            if matched is not None:
                matched_points.append(matched)
        
        # Flush remaining points
        for point in self.map_matcher.flush():
            matched_points.append(point)
        
        return matched_points

    @staticmethod
    def _count_compressed_points(compressed_segments: List[CompressedSegment]) -> int:
        """
        Count the total number of points in the compressed representation.
        
        Args:
            compressed_segments: List of CompressedSegment objects.
            
        Returns:
            Total point count in compressed representation.
        """
        total = 0
        for seg in compressed_segments:
            if seg.segment_type == "stop":
                # CompressedStop is represented as 1 point (centroid)
                total += 1
            elif seg.segment_type == "move":
                data = seg.compressed_data
                if isinstance(data, list):  # List[Point] from SQUISH
                    total += len(data)
                elif isinstance(data, dict):  # HYSOC-N: {trace_result, retained_points}
                    # Count the retained geographic points
                    retained = data.get('retained_points', [])
                    total += len(retained)
                elif isinstance(data, Move):  # Uncompressed move
                    total += len(data.points)
                else:  # Unknown format, estimate conservatively
                    total += 2  # At least start/end
        return total

    def get_compression_summary(self) -> str:
        """
        Generate a human-readable summary of compression settings.
        
        Returns:
            String describing the compression configuration.
        """
        lines = [
            "=" * 60,
            "HYSOC Compression Configuration",
            "=" * 60,
            f"Segmentation: STEP (ε={self.config.stop_max_eps_meters}m, T={self.config.stop_min_duration_seconds}s)",
            f"Stop Compression: {'Enabled' if self.config.compress_stops else 'Disabled'}",
            f"Move Compression: {self.config.move_compression_strategy.value.upper()}",
        ]
        
        if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
            lines.append(f"  - SquishCompressor with capacity {self.config.squish_capacity}")
        else:
            lines.append(f"  - TraceCompressor (Network-Semantic)")
            lines.append(f"  - Map Matching: {'Enabled' if self.config.enable_map_matching else 'Disabled'}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
