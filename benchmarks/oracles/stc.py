from typing import List

from hysoc.core.point import Point
from hysoc.core.segment import Segment


class STCOracle:
    """
    Offline Semantic Trajectory Compression (STC) Oracle.
    Compresses a map-matched trajectory segment by preserving only the sequence
    of reference points where the semantic mobility channel (road_id) changes,
    plus the final destination point.
    """

    def __init__(self):
        pass

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

        points = segment.points
        if len(points) <= 1:
            return list(points)

        compressed = []
        current_road = None

        for idx, point in enumerate(points):
            # The first point is always included as the origin of the chunk sequence.
            if idx == 0:
                compressed.append(point)
                current_road = point.road_id
                continue

            # The last point is always included as the destination.
            if idx == len(points) - 1:
                if compressed[-1] != point:
                    compressed.append(point)
                continue

            # If the road_id changes, we've entered a new semantic channel.
            # Add the first point of this new channel.
            if point.road_id != current_road:
                compressed.append(point)
                current_road = point.road_id

        return compressed
