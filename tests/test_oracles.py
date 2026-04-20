import sys
from pathlib import Path

import pytest
from datetime import datetime, timedelta

# Ensure the project root is on sys.path so `benchmarks.*` imports work.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from core.point import Point
from core.segment import Stop, Move
from oracle.oracleG import OracleG

@pytest.fixture
def synthetic_trajectory():
    """
    Creates a simple trajectory with two spatially separated stops:
    - 10 points at Stop A (0,0) over 10 minutes
    - 4 sparse "move" points bridging a ~110m gap (steps of ~55m >> max_eps=50)
    - 10 points at Stop B (0.001, 0.001) ~157m away over 10 minutes

    The move steps are deliberately larger than max_eps so the density
    backend (stss_manual) does not chain the two stops into one cluster.
    This keeps sklearn OPTICS (xi=0.02) and the DBSCAN-like manual backend
    agreeing on the ground-truth segment count.
    """
    points = []
    start_time = datetime(2024, 1, 1, 12, 0, 0)

    for i in range(10):
        points.append(Point(
            lat=0.0,
            lon=0.0,
            timestamp=start_time + timedelta(minutes=i),
            obj_id="obj1"
        ))

    # Sparse move: 2 intermediate points at ~1/3 and 2/3 of the 157 m gap,
    # yielding ~52 m between consecutive positions (above max_eps=50 m).
    for i, frac in enumerate((1.0 / 3.0, 2.0 / 3.0), start=1):
        delta = 0.001 * frac
        points.append(Point(
            lat=delta,
            lon=delta,
            timestamp=start_time + timedelta(minutes=10+i),
            obj_id="obj1"
        ))

    for i in range(10):
        points.append(Point(
            lat=0.001,
            lon=0.001,
            timestamp=start_time + timedelta(minutes=13+i),
            obj_id="obj1"
        ))

    return points

def test_stss_sklearn(synthetic_trajectory):
    # max_eps 50m. 0->0.001 deg is approx 111m. So A and B are distinct.
    # min_duration 60s. Points are 1 min apart.
    oracle = OracleG(min_samples=3, max_eps=50.0, min_duration_seconds=120.0)
    segments = oracle.process(synthetic_trajectory)
    
    # Expect: Stop A, Move, Stop B
    # Note: The move points might be noisy or part of moves.
    # Check logical structure
    
    stops = [s for s in segments if isinstance(s, Stop)]
    moves = [s for s in segments if isinstance(s, Move)]
    
    assert len(stops) >= 2
    
    # First stop should be near 0,0
    assert abs(stops[0].centroid.lat - 0.0) < 0.0001
    assert abs(stops[0].centroid.lon - 0.0) < 0.0001
    
    # Last stop should be near 0.001, 0.001
    assert abs(stops[-1].centroid.lat - 0.001) < 0.0001
    assert abs(stops[-1].centroid.lon - 0.001) < 0.0001

def test_stss_manual(synthetic_trajectory):
    oracle = OracleG(min_samples=3, max_eps=50.0, min_duration_seconds=120.0, backend="manual")
    segments = oracle.process(synthetic_trajectory)
    
    stops = [s for s in segments if isinstance(s, Stop)]
    
    assert len(stops) >= 2
    
    assert abs(stops[0].centroid.lat - 0.0) < 0.0001
    assert abs(stops[0].centroid.lon - 0.0) < 0.0001
    
    assert abs(stops[-1].centroid.lat - 0.001) < 0.0001
    assert abs(stops[-1].centroid.lon - 0.001) < 0.0001

def test_manual_parity(synthetic_trajectory):
    """
    Ensure both oracles produce similar results on simple data.
    """
    oracle1 = OracleG(min_samples=2, max_eps=50.0)
    oracle2 = OracleG(min_samples=2, max_eps=50.0, backend="manual")
    
    segs1 = oracle1.process(synthetic_trajectory)
    segs2 = oracle2.process(synthetic_trajectory)
    
    stops1 = [s for s in segs1 if isinstance(s, Stop)]
    stops2 = [s for s in segs2 if isinstance(s, Stop)]
    
    assert len(stops1) == len(stops2)
