"""
HYSOC Core: Compression Types and Configuration

Contains the shared types and data structures for the HYSOC compression pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from enum import Enum
from datetime import datetime

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move
from hysoc.modules.move_compression.trace import TraceConfig
from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.constants.segmentation_defaults import STOP_MAX_EPS_METERS, STOP_MIN_DURATION_SECONDS
from hysoc.constants.hysoc_defaults import HYSOC_DEFAULT_COMPRESS_STOPS


class CompressionStrategy(Enum):
    """Enum for move compression strategies."""
    GEOMETRIC = "geometric"  # HYSOC-G
    NETWORK_SEMANTIC = "network_semantic"  # HYSOC-N


@dataclass
class HYSOCConfig:
    """Configuration for the unified HYSOC pipeline."""
    move_compression_strategy: CompressionStrategy = CompressionStrategy.GEOMETRIC
    
    stop_max_eps_meters: float = STOP_MAX_EPS_METERS
    stop_min_duration_seconds: float = STOP_MIN_DURATION_SECONDS
    
    compress_stops: bool = HYSOC_DEFAULT_COMPRESS_STOPS
    
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
