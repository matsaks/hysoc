"""STC offline network-semantic move compressor (Oracle-N Module III)."""

from core.compression import SegmentResult
from core.point import Point

# Per road traversal: road_id (4) + entry_time (8) + exit_time (8).
_BYTES_ROAD_TRAVERSAL: int = 20


class STCOracle:
    """Offline Semantic Trajectory Compression oracle for the network track."""

    def __init__(self):
        pass

    def compress(self, points: list[Point]) -> SegmentResult:
        """Compress a map-matched point sequence using STC."""
        if not points:
            return SegmentResult(
                kind="move",
                start_time=None,
                end_time=None,
                keypoints=[],
                encoded_bytes=0,
            )

        keypoints = self._select_keypoints(points)
        return SegmentResult(
            kind="move",
            start_time=points[0].timestamp,
            end_time=points[-1].timestamp,
            keypoints=keypoints,
            encoded_bytes=max(0, len(keypoints) - 1) * _BYTES_ROAD_TRAVERSAL,
        )

    def _select_keypoints(self, points: list[Point]) -> list[Point]:
        """Return the road-transition entry points plus the final exit point."""
        if len(points) <= 1:
            return list(points)

        compressed: list[Point] = []
        current_road = None

        for idx, point in enumerate(points):
            if idx == 0:
                compressed.append(point)
                current_road = point.road_id
                continue

            if idx == len(points) - 1:
                if compressed[-1].timestamp != point.timestamp:
                    compressed.append(point)
                continue

            if point.road_id != current_road:
                compressed.append(point)
                current_road = point.road_id

        return compressed
