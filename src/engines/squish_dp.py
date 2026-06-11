"""Hybrid SQUISH plus Douglas-Peucker move compressor (HYSOC-G Module III)."""

from __future__ import annotations

from typing import List, Optional

from core.compression import BYTES_PER_POINT, SegmentResult
from core.point import Point
from core.segment import Move
from core.squish_dp_config import HybridSquishDPConfig
from engines.dp import DouglasPeuckerCompressor
from engines.squish import SquishCompressor


class HybridSquishDPCompressor:
    """Hybrid move compressor: SQUISH for long segments, DP for short segments."""

    def __init__(self, config: HybridSquishDPConfig = HybridSquishDPConfig()):
        if config.capacity < 3:
            raise ValueError("capacity must be >= 3")
        self.config = config
        self._squish = SquishCompressor(capacity=config.capacity)

    def compress(
        self,
        points: List[Point],
        *,
        capacity: Optional[int] = None,
        dp_epsilon_meters: Optional[float] = None,
    ) -> List[Point]:
        """Compress a single move segment to a list of keypoints."""
        if not points:
            return []

        cap = capacity if capacity is not None else self.config.capacity
        dp_eps = dp_epsilon_meters if dp_epsilon_meters is not None else self.config.dp_epsilon_meters

        # Below capacity SQUISH would keep everything, so run DP alone.
        if len(points) <= cap:
            dp = DouglasPeuckerCompressor(epsilon_meters=dp_eps)
            return dp.compress(points)

        squish_points = self._squish.compress(points, capacity=cap)

        if not self.config.dp_refine_when_evictions:
            return squish_points

        dp = DouglasPeuckerCompressor(epsilon_meters=dp_eps)
        return dp.compress(squish_points)

    def compress_segment(self, move: Move) -> SegmentResult:
        """Compress a Move segment and wrap the keypoints as a SegmentResult."""
        keypoints = self.compress(move.points)
        return SegmentResult(
            kind="move",
            start_time=move.start_time,
            end_time=move.end_time,
            keypoints=keypoints,
            encoded_bytes=len(keypoints) * BYTES_PER_POINT,
        )
