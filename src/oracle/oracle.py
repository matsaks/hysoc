"""Offline oracle baseline orchestrators (Oracle-G and Oracle-N)."""

from __future__ import annotations

from dataclasses import dataclass

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.hysoc_defaults import HYSOC_DEFAULT_COMPRESS_STOPS
from constants.segmentation_defaults import (
    STSS_MAX_EPS_METERS,
    STSS_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from constants.stop_compression_defaults import (
    STOP_COMPRESSION_DEFAULT_STRATEGY,
    StopCompressionStrategy,
)
from core.compression import (
    BYTES_PER_POINT,
    CompressionStrategy,
    SegmentResult,
    TrajectoryResult,
)
from core.point import Point
from core.segment import Move, Segment, Stop
from engines.dp import DouglasPeuckerCompressor
from engines.stc import STCOracle
from engines.stss import STSSOracle
from engines.stop_compressor import StopCompressor


@dataclass
class OracleBaseConfig:
    """Shared oracle pipeline configuration (Module I and II)."""
    stss_min_samples: int = STSS_MIN_SAMPLES
    stss_max_eps_meters: float = STSS_MAX_EPS_METERS
    stss_min_duration_seconds: float = STSS_MIN_DURATION_SECONDS
    compress_stops: bool = HYSOC_DEFAULT_COMPRESS_STOPS
    stop_compression_strategy: StopCompressionStrategy = STOP_COMPRESSION_DEFAULT_STRATEGY
    stop_preserve_road_id: bool = False


@dataclass
class OracleGConfig(OracleBaseConfig):
    """Oracle-G configuration: Module III via Douglas-Peucker."""
    dp_epsilon_meters: float = DP_DEFAULT_EPSILON_METERS


@dataclass
class OracleNConfig(OracleBaseConfig):
    """Oracle-N configuration: Module III via STC."""
    stop_preserve_road_id: bool = True


class _OracleBase:
    """Module I + II wiring and batch dispatch.  Subclasses provide Module III."""

    strategy: CompressionStrategy

    def __init__(self, config: OracleGConfig | OracleNConfig):
        self.config = config
        self.segmenter = STSSOracle(
            min_samples=config.stss_min_samples,
            max_eps=config.stss_max_eps_meters,
            min_duration_seconds=config.stss_min_duration_seconds,
        )
        self.stop_compressor = StopCompressor(
            config.stop_compression_strategy,
            preserve_road_id=config.stop_preserve_road_id,
        )

    def compress(
        self,
        points: list[Point],
        object_id: str | None = None,
    ) -> TrajectoryResult:
        oid = object_id if object_id is not None else (points[0].obj_id if points else "")
        if not points:
            return TrajectoryResult(
                object_id=oid,
                original_points=points,
                segments=[],
                strategy=self.strategy,
            )

        raw_segments = self.segmenter.process(points)
        segments = self._route(raw_segments)

        return TrajectoryResult(
            object_id=oid,
            original_points=points,
            segments=segments,
            strategy=self.strategy,
        )

    def _compress_move(self, seg: Move) -> SegmentResult:
        raise NotImplementedError

    def _route(self, segments: list[Segment]) -> list[SegmentResult]:
        out: list[SegmentResult] = []
        for s in segments:
            if isinstance(s, Stop):
                out.append(
                    self.stop_compressor.compress_segment(s)
                    if self.config.compress_stops
                    else StopCompressor.passthrough_segment(s)
                )
            elif isinstance(s, Move):
                out.append(self._compress_move(s))
        return out


class OracleGCompressor(_OracleBase):
    """Oracle-G: offline STSS segmentation + Douglas-Peucker move compression."""

    strategy = CompressionStrategy.GEOMETRIC

    def __init__(self, config: OracleGConfig | None = None):
        super().__init__(config or OracleGConfig())
        self.move_compressor = DouglasPeuckerCompressor(self.config.dp_epsilon_meters)

    def _compress_move(self, seg: Move) -> SegmentResult:
        keypoints = self.move_compressor.compress(seg.points)
        return SegmentResult(
            kind="move",
            start_time=seg.start_time,
            end_time=seg.end_time,
            keypoints=keypoints,
            encoded_bytes=len(keypoints) * BYTES_PER_POINT,
        )


class OracleNCompressor(_OracleBase):
    """Oracle-N: offline STSS segmentation + STC move compression."""

    strategy = CompressionStrategy.NETWORK_SEMANTIC

    def __init__(self, config: OracleNConfig | None = None):
        super().__init__(config or OracleNConfig())
        self.move_compressor = STCOracle()

    def _compress_move(self, seg: Move) -> SegmentResult:
        return self.move_compressor.compress(seg.points)
