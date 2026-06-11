"""HYSOC-G and HYSOC-N online compression orchestrators."""

from __future__ import annotations

from core.compression import (
    CompressionStrategy,
    HYSOCGConfig,
    HYSOCNConfig,
    SegmentResult,
    TrajectoryResult,
)
from core.point import Point
from core.segment import Move, Segment, Stop
from engines.squish_dp import HybridSquishDPCompressor
from engines.step import STEPSegmenter
from engines.stop_compressor import StopCompressor
from engines.trace import TraceCompressor


class _HYSOCBase:
    """Module I + II wiring and streaming dispatch.  Subclasses provide Module III."""

    strategy: CompressionStrategy

    def __init__(self, config: HYSOCGConfig | HYSOCNConfig):
        self.config = config
        self.segmenter = STEPSegmenter(
            max_eps=config.stop_max_eps_meters,
            min_duration_seconds=config.stop_min_duration_seconds,
        )
        self.stop_compressor = StopCompressor(
            config.stop_compression_strategy,
            preserve_road_id=config.stop_preserve_road_id,
        )

    def process_point(self, point: Point) -> list[SegmentResult]:
        prepared = self._prepare(point)
        if prepared is None:
            return []
        return self._route(self.segmenter.process_point(prepared))

    def flush(self) -> list[SegmentResult]:
        out: list[SegmentResult] = []
        for p in self._flush_pre():
            out.extend(self._route(self.segmenter.process_point(p)))
        out.extend(self._route(self.segmenter.flush()))
        return out

    def compress(
        self,
        points: list[Point],
        object_id: str | None = None,
    ) -> TrajectoryResult:
        segments: list[SegmentResult] = []
        for p in points:
            segments.extend(self.process_point(p))
        segments.extend(self.flush())

        oid = object_id if object_id is not None else (
            points[0].obj_id if points else ""
        )
        return TrajectoryResult(
            object_id=oid,
            original_points=points,
            segments=segments,
            strategy=self.strategy,
        )

    def _prepare(self, point: Point) -> Point | None:
        return point

    def _flush_pre(self) -> list[Point]:
        return []

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


class HYSOCGCompressor(_HYSOCBase):
    """HYSOC-G: Module III via SQUISH + Douglas-Peucker.  No map matching, no H."""

    strategy = CompressionStrategy.GEOMETRIC

    def __init__(self, config: HYSOCGConfig | None = None):
        super().__init__(config or HYSOCGConfig())
        self.move_compressor = HybridSquishDPCompressor(self.config.move_config)

    def _compress_move(self, seg: Move) -> SegmentResult:
        return self.move_compressor.compress_segment(seg)


class HYSOCNCompressor(_HYSOCBase):
    """HYSOC-N: Module III via TRACE; points must carry road_id on entry."""

    strategy = CompressionStrategy.NETWORK_SEMANTIC

    def __init__(
        self,
        config: HYSOCNConfig | None = None,
        trace_compressor: TraceCompressor | None = None,
    ):
        super().__init__(config or HYSOCNConfig())
        self.trace_compressor = trace_compressor or TraceCompressor(
            self.config.trace_config
        )

    def _compress_move(self, seg: Move) -> SegmentResult:
        return self.trace_compressor.compress(seg.points)
