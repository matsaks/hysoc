import unittest
from datetime import datetime
from hysoc.core.point import Point
from hysoc.modules.stop_compression.compressor import StopCompressor

class TestStopCompression(unittest.TestCase):
    def test_compress_centroid_and_structure(self):
        t1 = datetime(2023, 1, 1, 12, 0, 0)
        t2 = datetime(2023, 1, 1, 12, 0, 10)
        t3 = datetime(2023, 1, 1, 12, 0, 20)
        
        points = [
            Point(lat=10.0, lon=20.0, timestamp=t1, obj_id="1"),
            Point(lat=12.0, lon=22.0, timestamp=t2, obj_id="1"),
            Point(lat=11.0, lon=21.0, timestamp=t3, obj_id="1")
        ]
        
        compressor = StopCompressor()
        result = compressor.compress(points)
        
        # Average lat: (10+12+11)/3 = 33/3 = 11.0
        # Average lon: (20+22+21)/3 = 63/3 = 21.0
        
        self.assertEqual(result.centroid.lat, 11.0)
        self.assertEqual(result.centroid.lon, 21.0)
        self.assertEqual(result.centroid.obj_id, "1")
        
        # Check start and end times
        self.assertEqual(result.start_time, t1)
        self.assertEqual(result.end_time, t3)

if __name__ == '__main__':
    unittest.main()
