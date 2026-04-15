"""Default parameters for the Douglas-Peucker offline geometric oracle."""

from __future__ import annotations

# Spatial tolerance for the Ramer-Douglas-Peucker simplification (meters).
# Chosen as the baseline epsilon from the epsilon-sweep experiments.
DP_DEFAULT_EPSILON_METERS: float = 15.0
