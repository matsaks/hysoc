"""STOP/MOVE segmentation defaults shared by STEP and STSS."""

from __future__ import annotations

# Stay-point distance threshold (metres).
STOP_MAX_EPS_METERS: float = 15.0
STSS_MAX_EPS_METERS: float = 15.0

# Minimum dwell duration for a Stop (seconds).
STOP_MIN_DURATION_SECONDS: float = 30.0
STSS_MIN_DURATION_SECONDS: float = 30.0

# OPTICS density parameter for STSS, max(5, round(T * 0.5)).
STSS_MIN_SAMPLES: int = 15
