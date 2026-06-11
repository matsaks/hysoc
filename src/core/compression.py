"""Shared types and configuration for the HYSOC compression pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from constants.hysoc_defaults import HYSOC_DEFAULT_COMPRESS_STOPS
from constants.segmentation_defaults import STOP_MAX_EPS_METERS, STOP_MIN_DURATION_SECONDS
from constants.stop_compression_defaults import (
    StopCompressionStrategy,
    STOP_COMPRESSION_DEFAULT_STRATEGY,
)
from core.point import Point
from core.squish_dp_config import HybridSquishDPConfig
from core.trace_config import TraceConfig

# Byte cost of one raw GPS fix: lat + lon (float64) + timestamp (int64).
BYTES_PER_POINT: int = 24


class CompressionStrategy(Enum):
    """Move-segment compression strategy."""
    GEOMETRIC = "geometric"
    NETWORK_SEMANTIC = "network_semantic"


@dataclass
class HYSOCBaseConfig:
    """Shared HYSOC pipeline configuration (Module I and II)."""
    stop_max_eps_meters: float = STOP_MAX_EPS_METERS
    stop_min_duration_seconds: float = STOP_MIN_DURATION_SECONDS
    compress_stops: bool = HYSOC_DEFAULT_COMPRESS_STOPS
    stop_compression_strategy: StopCompressionStrategy = STOP_COMPRESSION_DEFAULT_STRATEGY
    # Snap the centroid keypoint's road_id from the nearest raw cluster point.
    stop_preserve_road_id: bool = False


@dataclass
class HYSOCGConfig(HYSOCBaseConfig):
    """HYSOC-G configuration: Module III via SQUISH + Douglas-Peucker."""
    move_config: HybridSquishDPConfig = field(default_factory=HybridSquishDPConfig)


@dataclass
class HYSOCNConfig(HYSOCBaseConfig):
    """HYSOC-N configuration: Module III via TRACE."""
    stop_preserve_road_id: bool = True
    trace_config: TraceConfig = field(default_factory=TraceConfig)


@dataclass(frozen=True)
class SegmentResult:
    """Compressed representation of a single Stop or Move segment."""
    kind: Literal["stop", "move"]
    start_time: datetime
    end_time: datetime
    keypoints: list[Point]
    encoded_bytes: int
    e_factors: list | None = None
    v_factors: list | None = None


@dataclass
class TrajectoryResult:
    """Standardised output of any HYSOC or oracle compression run."""
    object_id: str
    original_points: list[Point]
    segments: list[SegmentResult]
    strategy: CompressionStrategy

    @property
    def keypoints(self) -> list[Point]:
        """Flat, chronologically-ordered reconstruction of the compressed trajectory."""
        result: list[Point] = []
        for seg in self.segments:
            result.extend(seg.keypoints)
        return result

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

    def stops(self) -> list[SegmentResult]:
        return [s for s in self.segments if s.kind == "stop"]

    def moves(self) -> list[SegmentResult]:
        return [s for s in self.segments if s.kind == "move"]
