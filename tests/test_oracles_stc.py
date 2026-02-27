import pytest
from datetime import datetime
from hysoc.core.point import Point
from hysoc.core.segment import Move
from benchmarks.oracles.stc import STCOracle


def test_stc_oracle_length():
    points = [
        Point(lat=0.0, lon=0.0, timestamp=datetime(2025, 1, 1, 0, 0, 0), obj_id="1", road_id="A"),
        Point(lat=0.1, lon=0.1, timestamp=datetime(2025, 1, 1, 0, 0, 10), obj_id="1", road_id="A"),
        Point(lat=0.2, lon=0.2, timestamp=datetime(2025, 1, 1, 0, 0, 20), obj_id="1", road_id="B"),
        Point(lat=0.3, lon=0.3, timestamp=datetime(2025, 1, 1, 0, 0, 30), obj_id="1", road_id="B"),
        Point(lat=0.4, lon=0.4, timestamp=datetime(2025, 1, 1, 0, 0, 40), obj_id="1", road_id="C"),
    ]
    move = Move(points=points)
    oracle = STCOracle()
    compressed = oracle.process(move)
    
    assert len(compressed) == 3
    assert compressed[0] == points[0] # Start of A
    assert compressed[1] == points[2] # Start of B
    assert compressed[2] == points[4] # Start of C and destination

def test_stc_oracle_last_is_not_new():
    points = [
        Point(lat=0.0, lon=0.0, timestamp=datetime(2025, 1, 1, 0, 0, 0), obj_id="1", road_id="A"),
        Point(lat=0.1, lon=0.1, timestamp=datetime(2025, 1, 1, 0, 0, 10), obj_id="1", road_id="A"),
        Point(lat=0.2, lon=0.2, timestamp=datetime(2025, 1, 1, 0, 0, 20), obj_id="1", road_id="B"),
        Point(lat=0.3, lon=0.3, timestamp=datetime(2025, 1, 1, 0, 0, 30), obj_id="1", road_id="B"),
    ]
    move = Move(points=points)
    oracle = STCOracle()
    compressed = oracle.process(move)
    
    # idx 0: compressed=[0], current=A
    # idx 1: pass
    # idx 2: compressed=[0, 2], current=B
    # idx 3: last pos. compressed[-1] is 2. 3 != 2. compressed=[0, 2, 3]
    assert len(compressed) == 3
    assert compressed[0] == points[0] # Start of A
    assert compressed[1] == points[2] # Start of B
    assert compressed[2] == points[3] # Destination

def test_stc_oracle_single_point():
    points = [Point(lat=0.0, lon=0.0, timestamp=datetime(2025, 1, 1, 0, 0, 0), obj_id="1", road_id="A")]
    move = Move(points=points)
    oracle = STCOracle()
    compressed = oracle.process(move)
    assert len(compressed) == 1

