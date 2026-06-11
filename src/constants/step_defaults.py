"""STEP segmentation hyperparameters."""

from __future__ import annotations

from math import sqrt

# Grid cell size factor: g = STEP_DEFAULT_GRID_FACTOR * D.
STEP_DEFAULT_GRID_FACTOR: float = sqrt(2) / 4.0
