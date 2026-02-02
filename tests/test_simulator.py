import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from hysoc.simulation import TrajectorySimulator

@pytest.fixture
def sample_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_traj.csv"
    p.write_text("latitude,longitude\n10.0,20.0\n10.1,20.1\n10.2,20.2")
    return p

def test_obj_id_extraction(sample_csv):
    sim = TrajectorySimulator(sample_csv)
    assert sim._obj_id == "test_traj"

    sim_explicit = TrajectorySimulator(sample_csv, obj_id="explicit_id")
    assert sim_explicit._obj_id == "explicit_id"

def test_stream_points(sample_csv):
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    sim = TrajectorySimulator(sample_csv, interval=0.1, start_time=start_time)
    
    # Patch sleep to speed up test
    with patch("time.sleep") as mock_sleep:
        points = list(sim.stream())
    
    assert len(points) == 3
    
    assert points[0].lat == 10.0
    assert points[0].timestamp == start_time
    
    assert points[1].lat == 10.1
    assert points[1].timestamp == start_time + timedelta(seconds=0.1)
    
    assert points[2].lat == 10.2
    assert points[2].timestamp == start_time + timedelta(seconds=0.2)
    
    assert mock_sleep.call_count == 3 

def test_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("lat,lon\n10,20") # Wrong headers
    
    sim = TrajectorySimulator(p)
    with pytest.raises(ValueError, match="must contain 'latitude' and 'longitude'"):
        list(sim.stream())
