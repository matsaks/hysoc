"""
HYSOC: Hybrid Trajectory Compression Framework

This module orchestrates the unified pipeline combining:
- Module I: Streaming Segmentation (STEP)
- Module II: Stop Compression (StopCompressor)
- Module III-A: Move Compression - Geometric (DouglasPeuckerCompressor)
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
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor
from hysoc.modules.move_compression.trace import TraceCompressor, TraceConfig
from hysoc.modules.map_matching.wrapper import MapMatchedStreamWrapper
from hysoc.modules.map_matching.matcher import OnlineMapMatcher
from hysoc.constants.geo_defaults import EARTH_RADIUS_M
from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS


class CompressionStrategy(Enum):
    """Enum for move compression strategies."""
    GEOMETRIC = "geometric"  # HYSOC-G
    NETWORK_SEMANTIC = "network_semantic"  # HYSOC-N


@dataclass
class HYSOCConfig:
    """Configuration for the unified HYSOC pipeline."""
    move_compression_strategy: CompressionStrategy = CompressionStrategy.GEOMETRIC
    
    stop_max_eps_meters: float = 25.0  
    stop_min_duration_seconds: float = 10.0  
    
    compress_stops: bool = True
    
    dp_epsilon_meters: float = DP_DEFAULT_EPSILON_METERS
    
    trace_config: TraceConfig = field(default_factory=TraceConfig)
    osm_graph: Optional[Any] = None  
    enable_map_matching: bool = False


@dataclass
class CompressedSegment:
    """Represents a compressed segment (Stop or Move)."""
    segment_type: str  
    original_segment: Union[Stop, Move]
    compressed_data: Any  
    compression_ratio: Optional[float] = None


@dataclass
class CompressedTrajectory:
    """Result of compressing a complete trajectory (for batch execution)."""
    original_points: List[Point]
    compressed_segments: List[CompressedSegment]
    total_original_points: int
    total_compressed_points: int
    overall_compression_ratio: float
    compression_strategy: CompressionStrategy
    timestamp: datetime = field(default_factory=datetime.now)

    def get_reconstructed_points(self) -> List[Point]:
        """
        Reconstructs the compressed trajectory as a unified, continuous sequence of points.
        This provides a savable trajectory where move points and stop centroids
        are connected in chronological order.
        """
        points = []
        for seg in self.compressed_segments:
            if seg.segment_type == "stop":
                if hasattr(seg.compressed_data, "centroid"):
                    points.append(seg.compressed_data.centroid)
                elif hasattr(seg.compressed_data, "points") and seg.compressed_data.points:
                    points.extend(seg.compressed_data.points)
            elif seg.segment_type == "move":
                data = seg.compressed_data
                if isinstance(data, list):
                    points.extend(data)
                elif isinstance(data, dict) and 'retained_points' in data:
                    points.extend(data['retained_points'])
                elif hasattr(data, "points") and data.points:
                    points.extend(data.points)
        
        # Ensure points are ordered chronologically
        points.sort(key=lambda p: p.timestamp)
        return points


class HYSOCCompressor:
    """
    Unified HYSOC trajectory compression orchestrator.
    
    Acts as a true streaming pipeline. As points flow in via `process_point`,
    they get map-matched, segmented, and then compressed block-by-block. 
    Memory is kept low as it does not perpetually buffer history.
    """

    def __init__(self, config: HYSOCConfig = HYSOCConfig()):
        self.config = config
        
        if config.move_compression_strategy == CompressionStrategy.NETWORK_SEMANTIC:
            if config.enable_map_matching and config.osm_graph is None:
                raise ValueError(
                    "NETWORK_SEMANTIC strategy with map matching requires osm_graph."
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
            self.move_compressor = DouglasPeuckerCompressor(epsilon_meters=config.dp_epsilon_meters)
        else:  # NETWORK_SEMANTIC
            self.move_compressor = TraceCompressor(config=config.trace_config)
        
        # Initialize optional map matcher
        self.map_matcher: Optional[OnlineMapMatcher] = None
        if config.enable_map_matching and config.osm_graph is not None:
            self.map_matcher = OnlineMapMatcher(config.osm_graph)
            
        # Metrics state for stream health
        self._total_points_in = 0
        self._total_points_compressed = 0

    def process_point(self, point: Point) -> List[CompressedSegment]:
        """
        Streaming pipeline: processes a single point, routes it through the 
        active pipeline, and emits full compressed segments if a segment boundary closes.
        """
        self._total_points_in += 1
        
        # 1. Pipeline Stage: Map Matching
        if self.map_matcher is not None:
            matched_point = self.map_matcher.process_point(point)
            if matched_point is None:
                return []  # Still buffering in matcher
            point = matched_point
            
        # 2. Pipeline Stage: Segmentation
        segments = self.segmenter.process_point(point)
        
        # 3. Pipeline Stage: Compression Execution
        # Execute batch compression automatically on cleanly detected chunks
        compressed = []
        for seg in segments:
            c_seg = self._compress_segment(seg)
            if c_seg is not None:
                compressed.append(c_seg)
                
        return compressed

    def flush(self) -> List[CompressedSegment]:
        """
        Terminate streaming sequence, flushing and compressing uncommitted buffers.
        """
        compressed = []
        
        # Flush map matcher buffers through segmenter
        if self.map_matcher is not None:
            for point in self.map_matcher.flush():
                segments = self.segmenter.process_point(point)
                for seg in segments:
                    c_seg = self._compress_segment(seg)
                    if c_seg is not None:
                        compressed.append(c_seg)
                        
        # Flush final remaining segment states
        for seg in self.segmenter.flush():
            c_seg = self._compress_segment(seg)
            if c_seg is not None:
                compressed.append(c_seg)
                
        return compressed

    def _extract_retained_points_from_trace(self, points: List[Point]) -> List[Point]:
        """Identifies retained velocity-change points for TRACE rendering."""
        if not points or len(points) < 2:
            return points
            
        retained_points = []
        gamma = self.config.trace_config.gamma
        
        def lat_lon_dist(p1: Point, p2: Point) -> float:
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
            if i == 0 or p.road_id != current_road_id:
                current_road_id = p.road_id
                retained_points.append(p)
                last_stored_speed = 0.0
                continue
                
            prev_p = points[i - 1]
            dist = lat_lon_dist(prev_p, p)
            time_diff = (p.timestamp - prev_p.timestamp).total_seconds()
            
            if time_diff > 0:
                current_speed = dist / time_diff
            else:
                current_speed = last_stored_speed
                
            if abs(current_speed - last_stored_speed) > gamma:
                retained_points.append(p)
                last_stored_speed = current_speed
                
        if retained_points and retained_points[-1] != points[-1]:
            retained_points.append(points[-1])
            
        return retained_points

    def _compress_segment(self, seg: Segment) -> Optional[CompressedSegment]:
        """Runs defined compression payload targeting the specified segment class."""
        if isinstance(seg, Stop):
            if self.config.compress_stops:
                compressed_data = self.stop_compressor.compress(seg.points)
                total_compressed = 1  
            else:
                compressed_data = seg
                total_compressed = len(seg.points)
                
            compression_ratio = (len(seg.points) - total_compressed) / len(seg.points) if len(seg.points) > 0 else 0.0
            self._total_points_compressed += total_compressed
            
            return CompressedSegment(
                segment_type="stop",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
            )
            
        elif isinstance(seg, Move):
            if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
                compressed_data = self.move_compressor.compress(seg.points)
                total_compressed = len(compressed_data)
            else:
                trace_result = self.move_compressor.compress(seg.points)
                retained_points = self._extract_retained_points_from_trace(seg.points)
                compressed_data = {
                    'trace_result': trace_result,
                    'retained_points': retained_points
                }
                total_compressed = len(retained_points)
                
            compression_ratio = (len(seg.points) - total_compressed) / len(seg.points) if len(seg.points) > 0 else 0.0
            self._total_points_compressed += total_compressed
            
            return CompressedSegment(
                segment_type="move",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
            )
            
        return None

    def compress(self, points: List[Point]) -> CompressedTrajectory:
        """
        Legacy Batch Wrapper. Pushes all points structurally through the streaming
        pipeline and collects result representations.
        """
        # Reset modules
        self.__init__(self.config)
        
        compressed_segments = []
        for point in points:
            compressed_segments.extend(self.process_point(point))
            
        compressed_segments.extend(self.flush())
        
        ratio = 0.0
        if self._total_points_in > 0:
            ratio = (self._total_points_in - self._total_points_compressed) / self._total_points_in
            
        return CompressedTrajectory(
            original_points=points,
            compressed_segments=compressed_segments,
            total_original_points=self._total_points_in,
            total_compressed_points=self._total_points_compressed,
            overall_compression_ratio=ratio,
            compression_strategy=self.config.move_compression_strategy,
        )

    def get_compression_summary(self) -> str:
        lines = [
            "=" * 60,
            "HYSOC Compression Configuration",
            "=" * 60,
            f"Segmentation: STEP (ε={self.config.stop_max_eps_meters}m, T={self.config.stop_min_duration_seconds}s)",
            f"Stop Compression: {'Enabled' if self.config.compress_stops else 'Disabled'}",
            f"Move Compression: {self.config.move_compression_strategy.value.upper()}",
        ]
        
        if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
            lines.append(f"  - DouglasPeuckerCompressor with epsilon {self.config.dp_epsilon_meters}m")
        else:
            lines.append(f"  - TraceCompressor (Network-Semantic)")
            lines.append(f"  - Map Matching: {'Enabled' if self.config.enable_map_matching else 'Disabled'}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
