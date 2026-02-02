import pytest
from datetime import datetime, timedelta
from hysoc.core.point import Point
from hysoc.core.segment import Stop, Move
from benchmarks.oracles.stss_sklearn import STSSOracleSklearn
from benchmarks.oracles.stss_manual import STSSOracleManual

@pytest.fixture
def synthetic_trajectory():
    """
    Creates a simple trajectory:
    - 10 points at Stop A (0,0)
    - 10 points Moving to Stop B
    - 10 points at Stop B (0.001, 0.001) ~111m away
    Total 30 points.
    """
    points = []
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    
    # Stop A: 0.0, 0.0 for 10 mins (1 min interval)
    for i in range(10):
        points.append(Point(
            lat=0.0,
            lon=0.0,
            timestamp=start_time + timedelta(minutes=i),
            obj_id="obj1"
        ))
    
    # Move: 0.0 -> 0.001
    # 10 steps
    for i in range(1, 11):
        delta = 0.001 * (i / 10)
        points.append(Point(
            lat=delta,
            lon=delta,
            timestamp=start_time + timedelta(minutes=10+i),
            obj_id="obj1"
        ))

    # Stop B: 0.001, 0.001 for 10 mins
    for i in range(10):
        points.append(Point(
            lat=0.001,
            lon=0.001,
            timestamp=start_time + timedelta(minutes=21+i),
            obj_id="obj1"
        ))
        
    return points

def test_stss_sklearn(synthetic_trajectory):
    # max_eps 50m. 0->0.001 deg is approx 111m. So A and B are distinct.
    # min_duration 60s. Points are 1 min apart.
    oracle = STSSOracleSklearn(min_samples=3, max_eps=50.0, min_duration_seconds=120.0)
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
    oracle = STSSOracleManual(min_samples=3, max_eps=50.0, min_duration_seconds=120.0)
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
    oracle1 = STSSOracleSklearn(min_samples=2, max_eps=50.0)
    oracle2 = STSSOracleManual(min_samples=2, max_eps=50.0)
    
    segs1 = oracle1.process(synthetic_trajectory)
    segs2 = oracle2.process(synthetic_trajectory)
    
    stops1 = [s for s in segs1 if isinstance(s, Stop)]
    stops2 = [s for s in segs2 if isinstance(s, Stop)]
    
    assert len(stops1) == len(stops2)
