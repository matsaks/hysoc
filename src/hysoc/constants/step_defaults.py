"""STEP (segmentation) default hyperparameters that are not part of STSS defaults."""

from __future__ import annotations

from math import sqrt

# STEP uses g = (sqrt(2)/4) * D when grid_size_meters is not provided.
STEP_DEFAULT_GRID_FACTOR: float = sqrt(2) / 4.0

