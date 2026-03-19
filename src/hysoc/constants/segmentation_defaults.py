"""
Default parameters for behavioral stop/move segmentation.

These are intended to keep experiments consistent across scripts, demos, and benchmarks.
"""

from __future__ import annotations

# Distance threshold for stop/stay-point neighborhood checks (meters).
STOP_MAX_EPS_METERS: float = 20.0

# Minimum duration for a dwell to be treated as a Stop (seconds).
STOP_MIN_DURATION_SECONDS: float = 30.0

# Density parameter used by STSS' OPTICS/DBSCAN-like clustering to form candidate stop clusters.
# STEP segmentation does not have this parameter; it is kept here to reduce inconsistency wherever STSS is used.
STSS_MIN_SAMPLES: int = 5

