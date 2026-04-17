import unittest
from datetime import datetime, timedelta
import math
from engines.trace import TraceCompressor, TraceConfig
from core.point import Point

class TestTraceCompressor(unittest.TestCase):

    def setUp(self):
        self.config = TraceConfig(gamma=5.0) # threshold 5 m/s
        self.compressor = TraceCompressor(self.config)
        self.start_time = datetime(2023, 1, 1, 12, 0, 0)

    def test_speed_based_representation_empty(self):
        """Test with empty list of points."""
        self.assertEqual(self.compressor._speed_based_representation([]), [])

    def test_speed_based_representation_single_point(self):
        """Test with a single point creates one entry with 0 speed."""
        p1 = Point(lat=0.0, lon=0.0, timestamp=self.start_time, obj_id="O1", road_id=1)
        expected = [(1, 1, 0.0, 0.0)]
        self.assertEqual(self.compressor._speed_based_representation([p1]), expected)
        
    def test_speed_based_representation_constant_speed(self):
        """
        Test points moving at roughly constant speed.
        P1: t=0, speed 0 (start)
        P2: t=1s, dist ~11.1m (speed ~11.1 m/s) -> Change > 5.0 -> NEW ENTRY
        P3: t=2s, dist ~11.1m (speed ~11.1 m/s) -> Change < 5.0 -> NO ENTRY
        """
        points = []
        # P1
        points.append(Point(lat=0.0, lon=0.0, timestamp=self.start_time, obj_id="O1", road_id=1))
        
        # P2: 0.0001 deg lat difference is roughly 11.12 meters.
        # Time diff 1s -> Speed 11.12 m/s.
        points.append(Point(lat=0.0001, lon=0.0, timestamp=self.start_time + timedelta(seconds=1), obj_id="O1", road_id=1))
        
        # P3: Another 0.0001 deg lat difference.
        # Time diff 1s (total 2s) -> Speed 11.12 m/s.
        points.append(Point(lat=0.0002, lon=0.0, timestamp=self.start_time + timedelta(seconds=2), obj_id="O1", road_id=1))
        
        result = self.compressor._speed_based_representation(points)
        print("\nTest Constant Speed:")
        print("Points:", points)
        print("Result:", result)
        
        self.assertEqual(len(result), 2)
        
        # First entry: (1, 1, 0.0, 0.0)
        self.assertEqual(result[0], (1, 1, 0.0, 0.0))
        
        # Second entry
        r_road, r_dir, r_off, r_speed = result[1]
        self.assertEqual(r_road, 1)
        # Expected distance calculation
        # 1 deg lat = 111132.954m (approx, or R * pi / 180)
        # R = 6371000
        dist_expected = 6371000 * math.radians(0.0001)
        self.assertAlmostEqual(r_off, dist_expected, places=2)
        self.assertAlmostEqual(r_speed, dist_expected, places=2)
        
    def test_speed_based_representation_speed_change(self):
        """
        Test points where speed changes significantly.
        P1: start
        P2: speed 11 m/s -> entry
        P3: speed 22 m/s -> entry (diff > 5)
        """
        points = []
        points.append(Point(lat=0.0, lon=0.0, timestamp=self.start_time, obj_id="O1", road_id=1))
        
        # P2: speed ~11m/s
        points.append(Point(lat=0.0001, lon=0.0, timestamp=self.start_time + timedelta(seconds=1), obj_id="O1", road_id=1))
        
        # P3: Move 0.0002 deg in 1s -> ~22m/s
        # Total lat 0.0003
        points.append(Point(lat=0.0003, lon=0.0, timestamp=self.start_time + timedelta(seconds=2), obj_id="O1", road_id=1))
        
        result = self.compressor._speed_based_representation(points)
        print("\nTest Speed Change:")
        print("Points:", points)
        print("Result:", result)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], (1, 1, 0.0, 0.0))
        
        # P2 entry
        dist_p1_p2 = 6371000 * math.radians(0.0001)
        self.assertAlmostEqual(result[1][3], dist_p1_p2, places=2)
        
        # P3 entry
        dist_p2_p3 = 6371000 * math.radians(0.0002)
        offset_p3 = dist_p1_p2 + dist_p2_p3
        self.assertAlmostEqual(result[2][2], offset_p3, places=2) 
        self.assertAlmostEqual(result[2][3], dist_p2_p3, places=2) # Speed is distance over 1s

    def test_referential_compression_basic(self):
        """
        Test that referential compression works for exact matches.
        
        Scenario:
        1. T1: Moves along Road 1, 2, 3.
        2. T2: Identical movement (exact same Road IDs and Speeds).
        
        Expectation:
        - T1 is compressed as literals (since no references exist yet).
        - T1 is added to references.
        - T2 is compressed as a reference to T1.
        """
        # Create T1 points
        t1_points = [
            Point(lat=0.000, lon=0.0, timestamp=self.start_time, road_id=1, obj_id="O1"),
            Point(lat=0.001, lon=0.0, timestamp=self.start_time + timedelta(seconds=10), road_id=2, obj_id="O1"),
            Point(lat=0.002, lon=0.0, timestamp=self.start_time + timedelta(seconds=20), road_id=3, obj_id="O1"),
            Point(lat=0.003, lon=0.0, timestamp=self.start_time + timedelta(seconds=30), road_id=4, obj_id="O1"),
            Point(lat=0.004, lon=0.0, timestamp=self.start_time + timedelta(seconds=40), road_id=5, obj_id="O1"),
        ]
        
        # 1. Compress T1 -> Should be literals
        res_t1 = self.compressor.compress(t1_points)
        print("\nTest Ref Compression (T1):", res_t1)
        
        # T1 E-seq: [1, 2, 3, 4, 5]
        # Should be all literals because no history
        self.assertTrue(all(isinstance(x, int) for x in res_t1['E']))
        
        # Check that T1 was added as reference
        self.assertEqual(len(self.compressor.references), 1)
        ref_id = list(self.compressor.references.keys())[0]
        
        # 2. Create T2 identical to T1
        t2_points = [
            Point(lat=0.000, lon=0.0, timestamp=self.start_time + timedelta(minutes=5), road_id=1, obj_id="O2"),
            Point(lat=0.001, lon=0.0, timestamp=self.start_time + timedelta(minutes=5, seconds=10), road_id=2, obj_id="O2"),
            Point(lat=0.002, lon=0.0, timestamp=self.start_time + timedelta(minutes=5, seconds=20), road_id=3, obj_id="O2"),
            Point(lat=0.003, lon=0.0, timestamp=self.start_time + timedelta(minutes=5, seconds=30), road_id=4, obj_id="O2"),
            Point(lat=0.004, lon=0.0, timestamp=self.start_time + timedelta(minutes=5, seconds=40), road_id=5, obj_id="O2"),
        ]
        
        res_t2 = self.compressor.compress(t2_points)
        print("Test Ref Compression (T2):", res_t2)
        
        # T2 E-seq should be compressed using T1
        # With k=4, [1, 2, 3, 4] should match
        # Start at index 0, length 4 (or 5 if full match)
        # Expected: [(ref_id, 0, 5, None)] or similar
        
        # Note: The speed rep logic might produce slightly different items depending on gamma.
        # But assuming deterministic for identical input.
        
        found_ref_match = False
        for item in res_t2['E']:
            if isinstance(item, tuple) and len(item) == 4:
                # (ref_id, start, len, mismatch)
                r_id, start, length, mismatch = item
                if r_id == ref_id and start == 0 and length >= 4:
                    found_ref_match = True
        
        self.assertTrue(found_ref_match, "T2 should be compressed using reference T1")

    def test_referential_compression_partial_match(self):
        """
        Test that partial matches are found.
        
        T1: [1, 2, 3, 4, 5, 6]
        T2: [1, 2, 3, 4, 99, 99]
        
        Expect T2 to reference T1 for the first 4 elements.
        """
        # Config k=4
        
        # Construct T1 points (Roads 1..6)
        t1_points = []
        for i in range(1, 7):
             t1_points.append(Point(lat=i*0.001, lon=0, timestamp=self.start_time + timedelta(seconds=i*10), road_id=i, obj_id="O1"))
             
        self.compressor.compress(t1_points)
        
        # Construct T2 points (Roads 1..4, then 99, 100)
        t2_points = []
        for i in range(1, 5):
             t2_points.append(Point(lat=i*0.001, lon=0, timestamp=self.start_time + timedelta(minutes=5, seconds=i*10), road_id=i, obj_id="O2"))
             
        t2_points.append(Point(lat=0.050, lon=0, timestamp=self.start_time + timedelta(minutes=5, seconds=50), road_id=99, obj_id="O2"))
        t2_points.append(Point(lat=0.051, lon=0, timestamp=self.start_time + timedelta(minutes=5, seconds=60), road_id=100, obj_id="O2"))
        
        res_t2 = self.compressor.compress(t2_points)
        print("Test Partial Match (T2):", res_t2['E'])
        
        # Expectation:
        # Sequence: 1, 2, 3, 4, 99, 100
        # k=4 Match on [1,2,3,4]
        # mismatch at 99?
        # The compressor consumes mismatch in the tuple.
        # So we expect (ref_id, 0, 4, 99) and then literal 100.
        
        matched_tuple = None
        for item in res_t2['E']:
            if isinstance(item, tuple):
                matched_tuple = item
                break
        
        self.assertIsNotNone(matched_tuple)
        self.assertEqual(matched_tuple[2], 4) # length
        self.assertEqual(matched_tuple[3], 99) # mismatch value

    def test_reference_deletion(self):
        """
        Test that old references are removed.
        """
        # Set cleanup threshold high or manipulate decay to force deletion
        # Compressor config: decay_lambda = 0.9
        
        # Add Reference T1 at time 0
        t1_points = [Point(lat=0, lon=0, timestamp=self.start_time, road_id=1, obj_id="O1")]
        self.compressor.compress(t1_points)
        ref_id = list(self.compressor.references.keys())[0]
        
        # Move time forward significantly
        future_time = self.start_time + timedelta(hours=10) # 36000 seconds
        
        # Compress T2 at future time. This triggers _manage_references with current_time = future_time
        # T1's freshness will be 0.9 ^ 36000 ~ 0
        # It should be deleted.
        
        t2_points = [Point(lat=0, lon=0, timestamp=future_time, road_id=2, obj_id="O2")]
        self.compressor.compress(t2_points)
        
        # Check if T1 is gone
        self.assertNotIn(ref_id, self.compressor.references)
        
        # Check index cleanup
        # T1 had road_id=1. If we search for it, it should be empty.
        # Although k=4 and T1 has length 1, so it wasn't indexed?
        # TraceConfig default k=4. T1 length 1 -> No k-mers.
        # Wait, for this test to be valid for index cleanup, T1 needs length >= k.
        
        # Retry with longer T1
        self.compressor.references.clear()
        self.compressor.kmer_index.clear()
        
        t1_points = [Point(lat=i*0.001, lon=0, timestamp=self.start_time + timedelta(seconds=i), road_id=i, obj_id="O1") for i in range(10)]
        self.compressor.compress(t1_points)
        ref_id = list(self.compressor.references.keys())[0]
        
        # Verify index has entries
        self.assertTrue(len(self.compressor.kmer_index) > 0)
        
        # Trigger cleanup
        t2_points = [Point(lat=0, lon=0, timestamp=future_time, road_id=100, obj_id="O2")]
        self.compressor.compress(t2_points)
        
        self.assertNotIn(ref_id, self.compressor.references)
        # Verify index is empty (since T1 was the only ref)
        self.assertEqual(len(self.compressor.kmer_index), 0)

if __name__ == '__main__':
    unittest.main()
