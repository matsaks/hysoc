import unittest
from datetime import datetime, timedelta
import math
from hysoc.core.point import Point
from hysoc.metrics import calculate_compression_ratio, calculate_sed_stats, calculate_sed_error

class TestTrajectoryMetrics(unittest.TestCase):
    def setUp(self):
        self.start_time = datetime(2023, 1, 1, 12, 0, 0)

    def create_point(self, i, lat, lon):
        return Point(
            lat=lat,
            lon=lon,
            timestamp=self.start_time + timedelta(minutes=i),
            obj_id="obj1"
        )

    def test_compression_ratio(self):
        original = [self.create_point(i, 0, 0) for i in range(10)]
        compressed = [self.create_point(i, 0, 0) for i in range(0, 10, 9)] # 2 points: 0 and 9
        
        ratio = calculate_compression_ratio(original, compressed)
        self.assertEqual(ratio, 5.0) # 10 / 2 = 5
        
        ratio_empty = calculate_compression_ratio(original, [])
        self.assertEqual(ratio_empty, 1.0)

    def test_sed_zero(self):
        # Identical trajectories should have 0 error
        points = [self.create_point(i, float(i), float(i)) for i in range(3)]
        stats = calculate_sed_stats(points, points)
        self.assertEqual(stats['average_sed'], 0.0)
        self.assertEqual(stats['max_sed'], 0.0)
        self.assertEqual(stats['rmse'], 0.0)

    def test_sed_simple_interpolation(self):
        # 3 points in a line: (0,0,T0), (1,1,T1), (2,2,T2)
        # Compressed to: (0,0,T0), (2,2,T2)
        # T1 is exactly halfway between T0 and T2 (1 minute intervals)
        # Interpolation at T1 should account for (1,1).
        # Error should be 0 because it's a straight line.
        
        p0 = self.create_point(0, 0.0, 0.0)
        p1 = self.create_point(1, 1.0, 1.0)
        p2 = self.create_point(2, 2.0, 2.0)
        
        original = [p0, p1, p2]
        compressed = [p0, p2]
        
        stats = calculate_sed_stats(original, compressed)
        self.assertAlmostEqual(stats['average_sed'], 0.0)
        
    def test_sed_error_calculation(self):
        # Test the helper function directly with a known offset
        # Segment: (0,0, T0) -> (2,0, T2)
        # Point to test: (1, 1, T1)  <-- deviated by 1.0 in lat
        # Expected projection at T1: (1, 0)
        # Distance: sqrt((1-1)^2 + (1-0)^2) = 1.0
        
        p_start = self.create_point(0, 0.0, 0.0)
        p_end = self.create_point(2, 2.0, 0.0)
        p_test = self.create_point(1, 1.0, 1.0)
        
        error = calculate_sed_error(p_test, p_start, p_end)
        self.assertAlmostEqual(error, 1.0)
        
    def test_sed_stats_calculation(self):
        # Use previous example in stats
        # Original: p_start, p_test, p_end
        # Compressed: p_start, p_end
        # Errors: 0.0 (at start), 1.0 (at test), 0.0 (at end)
        # Average: (0+1+0)/3 = 0.333...
        # Max: 1.0
        # RMSE: sqrt((0^2 + 1^2 + 0^2)/3) = sqrt(1/3) = 0.577...
        
        p_start = self.create_point(0, 0.0, 0.0)
        p_end = self.create_point(2, 2.0, 0.0)
        p_test = self.create_point(1, 1.0, 1.0)
        
        original = [p_start, p_test, p_end]
        compressed = [p_start, p_end]
        
        stats = calculate_sed_stats(original, compressed)
        self.assertAlmostEqual(stats['average_sed'], 1.0/3.0)
        self.assertAlmostEqual(stats['max_sed'], 1.0)
        self.assertAlmostEqual(stats['rmse'], math.sqrt(1.0/3.0))

if __name__ == '__main__':
    unittest.main()
