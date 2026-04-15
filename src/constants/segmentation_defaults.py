"""
Default parameters for behavioral stop/move segmentation.

These are intended to keep experiments consistent across scripts, demos, and benchmarks.
"""

from __future__ import annotations

# Distance threshold for stop/stay-point neighborhood checks (meters).
STOP_MAX_EPS_METERS: float = 15.0
STSS_MAX_EPS_METERS: float = 15.0

# Minimum duration for a dwell to be treated as a Stop (seconds).
STOP_MIN_DURATION_SECONDS: float = 15.0
STSS_MIN_DURATION_SECONDS: float = 15.0

# Density parameter used by STSS' OPTICS/DBSCAN-like clustering to form candidate stop clusters.
# For a 1Hz dataset and a 15s stop duration, 15 points are theoretically expected.
# We set this to 7 to allow for up to 50% GPS point-drop loss while still strictly enforcing density.
STSS_MIN_SAMPLES: int = 7

