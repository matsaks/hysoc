import unittest
from datetime import datetime, timedelta

from core.point import Point
from engines.move_compression.dp import DouglasPeuckerCompressor
from engines.move_compression.hybrid_squish_dp import (
    HybridSquishDPCompressor,
    HybridSquishDPConfig,
)
from engines.move_compression.squish import SquishCompressor


class TestHybridSquishDPCompressor(unittest.TestCase):
    def setUp(self) -> None:
        self.start_time = datetime(2023, 1, 1, 12, 0, 0)

    def create_point(self, i: int, lat: float = 0.0, lon: float = 0.0) -> Point:
        return Point(
            lat=lat,
            lon=lon,
            timestamp=self.start_time + timedelta(seconds=i),
            obj_id="obj1",
        )

    def test_short_segment_uses_dp(self) -> None:
        # Horizontal line -> perpendicular distance to the start-end segment is 0.
        points = [self.create_point(i, lat=0.0, lon=float(i)) for i in range(6)]

        capacity = 10  # segment length <= capacity => "no evictions" branch
        dp_epsilon = 0.01

        expected = DouglasPeuckerCompressor(epsilon_meters=dp_epsilon).compress(points)

        compressor = HybridSquishDPCompressor(
            HybridSquishDPConfig(capacity=capacity, dp_epsilon_meters=dp_epsilon)
        )
        result = compressor.compress(points, capacity=capacity, dp_epsilon_meters=dp_epsilon)

        self.assertEqual(result, expected)

    def test_long_segment_uses_squish_without_refine(self) -> None:
        points = [self.create_point(i, lat=0.0, lon=float(i)) for i in range(8)]

        capacity = 5  # segment length > capacity => evictions branch
        dp_epsilon = 0.01

        expected = SquishCompressor(capacity=capacity).compress(points, capacity=capacity)

        compressor = HybridSquishDPCompressor(
            HybridSquishDPConfig(capacity=capacity, dp_epsilon_meters=dp_epsilon, dp_refine_when_evictions=False)
        )
        result = compressor.compress(points, capacity=capacity, dp_epsilon_meters=dp_epsilon)

        self.assertEqual(result, expected)

    def test_long_segment_dp_refine_optional(self) -> None:
        points = [self.create_point(i, lat=0.0, lon=float(i)) for i in range(8)]

        capacity = 5
        dp_epsilon = 0.01

        squish_points = SquishCompressor(capacity=capacity).compress(points, capacity=capacity)
        expected = DouglasPeuckerCompressor(epsilon_meters=dp_epsilon).compress(squish_points)

        compressor = HybridSquishDPCompressor(
            HybridSquishDPConfig(capacity=capacity, dp_epsilon_meters=dp_epsilon, dp_refine_when_evictions=True)
        )
        result = compressor.compress(points, capacity=capacity, dp_epsilon_meters=dp_epsilon)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

