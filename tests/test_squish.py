import unittest
from datetime import datetime, timedelta
from hysoc.core.point import Point
from hysoc.modules.move_compression.squish import SquishCompressor

class TestSquishCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = SquishCompressor(capacity=5)
        self.start_time = datetime(2023, 1, 1, 12, 0, 0)

    def create_point(self, i, lat, lon):
        return Point(
            lat=lat,
            lon=lon,
            timestamp=self.start_time + timedelta(minutes=i),
            obj_id="obj1"
        )

    def test_compress_empty(self):
        result = self.compressor.compress([])
        self.assertEqual(result, [])

    def test_compress_within_capacity(self):
        points = [self.create_point(i, float(i), float(i)) for i in range(4)]
        result = self.compressor.compress(points)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, points)

    def test_compress_simple_line(self):
        # A straight line should remove intermediate points if capacity is small
        # Points: (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)
        # Capacity is 5. Total 6 points. Should remove one.
        # All intermediate points have SED = 0 (perfect line).
        # Any of them could be removed.
        points = [self.create_point(i, float(i), float(i)) for i in range(6)]
        
        # Manually increase capacity to simplify test logic first
        compressor = SquishCompressor(capacity=3)
        result = compressor.compress(points)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], points[0])
        self.assertEqual(result[-1], points[-1])
        # Middle point should be one of the intermediate ones.
        # Since all have 0 error, any is fine. typically the ones added later might have less priority 
        # or pushed down relative to others, but with priority 0 it's stable.
        
    def test_compress_triangle(self):
        # Points: (0,0), (1,0.1), (2,2), (3,0.1), (4,0)
        # 0->1->2->3->4
        # 0 and 4 are start/end.
        # 2 is the peak (farthest from line 1-3).
        # 1 and 3 are close to the baseline.
        # If capacity=3, we expect to keep 0, 2, 4. Or maybe 0, 4 and one of 1/3?
        # Let's trace priorities:
        # P0(0,0), P1(1,0.1), P2(2,2), P3(3,0.1), P4(4,0)
        # T=0, 1, 2, 3, 4
        
        # Priorities when first calculated:
        # P1 (in 0-1-2): dist(P1, line P0-P2). P0(0,0), P2(2,2). Mid(1,1). P1(1,0.1). Error ~0.9
        # P2 (in 1-2-3): dist(P2, line P1-P3). P1(1,0.1), P3(3,0.1). Mid(2,0.1). P2(2,2). Error ~1.9
        # P3 (in 2-3-4): dist(P3, line P2-P4). P2(2,2), P4(4,0). Mid(3,1). P3(3,0.1). Error ~0.9
        
        # So P2 has highest priority. P1 and P3 have lower.
        # If capacity=3, we remove 2 points.
        # P1 and P3 should be removed. Result should be P0, P2, P4 ? 
        # Wait, if we remove P1:
        # P2 neighbors become P0, P3. 
        # P2 priority recomputed on P0...P3 line. P0(0,0), P3(3, 0.1). 
        # At T=2 (2/3 of way), pred is (2, 0.066). P2 is (2,2). Error ~1.93. Still high.
        
        # So yes, P2 should be kept.
        
        points = [
            self.create_point(0, 0.0, 0.0),
            self.create_point(1, 1.0, 0.1),
            self.create_point(2, 2.0, 2.0),
            self.create_point(3, 3.0, 0.1),
            self.create_point(4, 4.0, 0.0),
        ]
        
        compressor = SquishCompressor(capacity=3)
        result = compressor.compress(points)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], points[0])
        self.assertEqual(result[-1], points[-1])
        # Check if P2 is in the result
        self.assertTrue(any(p.lat == 2.0 and p.lon == 2.0 for p in result))

if __name__ == '__main__':
    unittest.main()
