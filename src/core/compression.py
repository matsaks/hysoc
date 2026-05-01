"""
HYSOC Core: Compression Types and Configuration

Contains the shared types and data structures for the HYSOC compression pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from core.point import Point
from core.trace_config import TraceConfig
from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.hysoc_defaults import HYSOC_DEFAULT_COMPRESS_STOPS
from constants.segmentation_defaults import STOP_MAX_EPS_METERS, STOP_MIN_DURATION_SECONDS
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from constants.stop_compression_defaults import StopCompressionStrategy, STOP_COMPRESSION_DEFAULT_STRATEGY

# Byte cost of one raw GPS fix: lat (float64=8) + lon (float64=8) + timestamp (int64=8).
BYTES_PER_POINT: int = 24


class CompressionStrategy(Enum):
    """Move-segment compression strategy."""
    GEOMETRIC = "geometric"
    NETWORK_SEMANTIC = "network_semantic"


@dataclass
class HYSOCConfig:
    """Configuration for the unified HYSOC pipeline."""
    move_compression_strategy: CompressionStrategy = CompressionStrategy.GEOMETRIC
    stop_compression_strategy: StopCompressionStrategy = STOP_COMPRESSION_DEFAULT_STRATEGY
    stop_max_eps_meters: float = STOP_MAX_EPS_METERS
    stop_min_duration_seconds: float = STOP_MIN_DURATION_SECONDS
    compress_stops: bool = HYSOC_DEFAULT_COMPRESS_STOPS
    squish_buffer_capacity: int = SQUISH_DEFAULT_CAPACITY
    dp_epsilon_meters: float = DP_DEFAULT_EPSILON_METERS
    trace_config: TraceConfig = field(default_factory=TraceConfig)
    osm_graph: Optional[Any] = None
    enable_map_matching: bool = False


@dataclass(frozen=True)
class SegmentResult:
    """
    Compressed representation of a single Stop or Move segment.

    keypoints     — reconstruction points for SED and trajectory rebuilding.
                    For stops: the centroid (one point).
                    For geometric moves: SQUISH/DP output.
                    For network moves: TRACE residual points.
    encoded_bytes — byte cost of the compressed representation.
                    For point-list strategies: len(keypoints) * BYTES_PER_POINT.
                    For TRACE: the actual encoding size from the compressor.
    """
    kind: Literal["stop", "move"]
    start_time: datetime
    end_time: datetime
    keypoints: list[Point]
    encoded_bytes: int


@dataclass
class TrajectoryResult:
    """
    Standardised output of any HYSOC or oracle compression run.

    All strategies — HYSOC-G, HYSOC-N, OracleDP, OracleSTC — produce a
    TrajectoryResult so that eval code operates on a single type.
    """
    object_id: str
    original_points: list[Point]
    segments: list[SegmentResult]
    strategy: CompressionStrategy

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @property
    def keypoints(self) -> list[Point]:
        """Flat, chronologically-ordered reconstruction of the compressed trajectory."""
        result: list[Point] = []
        for seg in self.segments:
            result.extend(seg.keypoints)
        return result

    # ------------------------------------------------------------------
    # Byte-based compression metrics
    # ------------------------------------------------------------------

    @property
    def original_bytes(self) -> int:
        return len(self.original_points) * BYTES_PER_POINT

    @property
    def encoded_bytes(self) -> int:
        return sum(s.encoded_bytes for s in self.segments)

    @property
    def compression_ratio(self) -> float:
        """original_bytes / encoded_bytes. Higher is better."""
        enc = self.encoded_bytes
        if enc == 0:
            return float("inf")
        return self.original_bytes / enc

    # ------------------------------------------------------------------
    # Segment filters
    # ------------------------------------------------------------------

    def stops(self) -> list[SegmentResult]:
        return [s for s in self.segments if s.kind == "stop"]

    def moves(self) -> list[SegmentResult]:
        return [s for s in self.segments if s.kind == "move"]
