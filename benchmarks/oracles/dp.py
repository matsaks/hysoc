from typing import List

from hysoc.constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from hysoc.core.point import Point
from hysoc.core.segment import Segment
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor


class DPOracle:
    """
    Offline Geometric Line Simplification Oracle utilizing the
    Ramer-Douglas-Peucker (DP) algorithm.
    """

    def __init__(self, epsilon_meters: float = DP_DEFAULT_EPSILON_METERS):
        """
        Initialize the DP Oracle with a spatial tolerance limit.

        Args:
            epsilon_meters: Tolerance deviation threshold in meters.
                Defaults to DP_DEFAULT_EPSILON_METERS (15.0 m), the baseline
                epsilon selected by the epsilon-sweep experiments.
        """
        self.compressor = DouglasPeuckerCompressor(epsilon_meters=epsilon_meters)

    def process(self, segment: Segment) -> List[Point]:
        """
        Compress a logical trajectory segment using Douglas-Peucker.

        Args:
            segment: The segment (usually a Move segment) to be compressed.

        Returns:
            A list of compressed points representing the simplified path.
        """
        if not segment.points:
            return []
            
        return self.compressor.compress(segment.points)
