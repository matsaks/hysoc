from typing import Iterator

from hysoc.core.point import Point
from hysoc.modules.map_matching.matcher import OnlineMapMatcher


class MapMatchedStreamWrapper:
    """
    Wraps an existing stream of GPS points and applies the OnlineMapMatcher to them.
    This dynamically injects the `road_id` onto points as they flow through the pipeline.
    """

    def __init__(self, point_stream: Iterator[Point], matcher: OnlineMapMatcher):
        """
        Args:
            point_stream: An iterator or generator that yields Point objects.
            matcher: An initialized OnlineMapMatcher instance.
        """
        self.point_stream = point_stream
        self.matcher = matcher

    def __iter__(self) -> Iterator[Point]:
        return self.stream()

    def stream(self) -> Iterator[Point]:
        """
        Yields points that have been processed by the map matcher.
        Due to the sliding window, initial points are delayed until the window fills.
        """
        for point in self.point_stream:
            matched_point = self.matcher.process_point(point)
            if matched_point is not None:
                yield matched_point

        # When the underlying stream ends, flush the remaining points in the matcher's buffer
        for point in self.matcher.flush():
            yield point
