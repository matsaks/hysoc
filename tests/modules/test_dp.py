from datetime import datetime, timedelta
import pytest

from hysoc.core.point import Point
from hysoc.modules.move_compression.dp import DouglasPeuckerCompressor

def create_straight_line(n_points: int, start_lat: float, start_lon: float, lat_step: float, lon_step: float) -> list[Point]:
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    points = []
    for i in range(n_points):
        lat = start_lat + (i * lat_step)
        lon = start_lon + (i * lon_step)
        t = base_time + timedelta(seconds=i*10)
        points.append(Point(lat=lat, lon=lon, timestamp=t, obj_id="test"))
    return points

def test_dp_straight_line():
    compressor = DouglasPeuckerCompressor(epsilon_meters=1.0)
    points = create_straight_line(10, 63.4, 10.4, 0.001, 0.001)
    
    compressed = compressor.compress(points)
    
    # DP should perfectly compress a mathematically straight line into just start and end points
    assert len(compressed) == 2
    assert compressed[0] == points[0]
    assert compressed[1] == points[-1]

def test_dp_significant_deviation():
    # Roughly ~111 meters per 0.001 degree of latitude. At lat 63, lon is about half.
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    points = [
        Point(lat=63.400, lon=10.400, timestamp=base_time, obj_id="test"),
        Point(lat=63.401, lon=10.400, timestamp=base_time + timedelta(seconds=10), obj_id="test"),
        Point(lat=63.402, lon=10.420, timestamp=base_time + timedelta(seconds=20), obj_id="test"), # Deviates MASSIVELY East (~1000m)
        Point(lat=63.403, lon=10.400, timestamp=base_time + timedelta(seconds=30), obj_id="test"),
        Point(lat=63.404, lon=10.400, timestamp=base_time + timedelta(seconds=40), obj_id="test")
    ]
    
    compressor = DouglasPeuckerCompressor(epsilon_meters=50.0)
    compressed = compressor.compress(points)
    
    assert len(compressed) == 3
    assert compressed[0] == points[0]
    assert compressed[1] == points[2]
    assert compressed[2] == points[-1]
    
def test_dp_tight_tolerance():
    # If tolerance is 0, practically everything is kept.
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    points = [
        Point(lat=63.400, lon=10.400, timestamp=base_time, obj_id="test"),
        Point(lat=63.401, lon=10.400, timestamp=base_time + timedelta(seconds=10), obj_id="test"),
        Point(lat=63.402, lon=10.402, timestamp=base_time + timedelta(seconds=20), obj_id="test"),
        Point(lat=63.403, lon=10.400, timestamp=base_time + timedelta(seconds=30), obj_id="test"),
        Point(lat=63.404, lon=10.400, timestamp=base_time + timedelta(seconds=40), obj_id="test")
    ]
    
    compressor = DouglasPeuckerCompressor(epsilon_meters=0.1) # extremely strict
    compressed = compressor.compress(points)
    
    # Retains all points since 0.1m is incredibly small
    assert len(compressed) == len(points)
