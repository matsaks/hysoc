from typing import List

from core.point import Point
from core.segment import Segment
from engines.stc import STCOracle as _STCCompressor


class OracleN:
    """
    Offline Network-Semantic Move-Compression Oracle utilizing the STC
    (Semantic Trajectory Compression) algorithm.

    Acts as a thin orchestrator on top of the STC engine; all road-id change
    detection logic lives in `engines/stc.py`.
    """

    def __init__(self):
        """Initialize the network-semantic oracle."""
        self.compressor = _STCCompressor()

    def process(self, segment: Segment) -> List[Point]:
        """
        Compress a logical trajectory segment using STC.

        Args:
            segment: The segment (usually a Move segment) to be compressed.

        Returns:
            A list of compressed points representing the semantic path.
        """
        if not segment.points:
            return []

        return self.compressor.process(segment)
