from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from hysoc.core.point import Point
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor
from hysoc.modules.move_compression.squish import SquishCompressor


@dataclass(frozen=True)
class HybridSquishDPConfig:
    """
    Configuration for Hybrid SQUISH + DP move compression (HYSOC-G hybrid).

    Strategy:
    - If the current move segment length <= buffer capacity, run DP on the full segment.
      (In this branch, SQUISH would never evict points, so DP can exploit the full set.)
    - If the move segment length > buffer capacity, run SQUISH with the fixed buffer.
    - Optionally (research switch), run DP as a second-pass refinement on the SQUISH survivors.
    """

    capacity: int = SQUISH_DEFAULT_CAPACITY
    dp_epsilon_meters: float = DP_DEFAULT_EPSILON_METERS
    dp_refine_when_evictions: bool = False


class HybridSquishDPCompressor:
    """
    Hybrid move compressor: SQUISH for long segments, DP for short segments.

    Notes on error interpretation:
    - SQUISH is capacity-based and optimizes SED-style intermediate error (internally).
    - DP here is the offline Ramer-Douglas-Peucker variant using perpendicular distance
      threshold in meters (not SED). We still expose `dp_epsilon_meters` so you can
      align experiments with the DP oracle baseline.
    """

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
        """
        Compress a single move segment.

        Args:
            points: Ordered GPS points belonging to one Move segment.
            capacity: Optional override of the fixed buffer capacity.
            dp_epsilon_meters: Optional override of DP's epsilon (meters).
        """
        if not points:
            return []

        cap = capacity if capacity is not None else self.config.capacity
        dp_eps = dp_epsilon_meters if dp_epsilon_meters is not None else self.config.dp_epsilon_meters

        # If we never exceed capacity, the SQUISH buffer contains the entire segment.
        # This exactly matches the "evictions_occurred == False" branch in the idea doc.
        if len(points) <= cap:
            dp = DouglasPeuckerCompressor(epsilon_meters=dp_eps)
            return dp.compress(points)

        squish_points = self._squish.compress(points, capacity=cap)

        if not self.config.dp_refine_when_evictions:
            return squish_points

        dp = DouglasPeuckerCompressor(epsilon_meters=dp_eps)
        return dp.compress(squish_points)

