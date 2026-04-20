"""
STEP (segmentation) default hyperparameters that are not part of STSS
defaults. See constants/segmentation_defaults.py for the shared (D, T).

Source
------
Sun et al., "Streaming Trajectory Segmentation Based on Stay-Point
Detection", DASFAA 2024. The grid cell size g is parameterised by an
integer n >= 1:

    g = (sqrt(2) / (4 * n)) * D

Larger n produces a finer grid, which enlarges the Confirmed region and
shrinks the Check region (more pruning, less exact distance work), at the
cost of a larger grid index. The paper uses n = 1 as the baseline.
We retain n = 1; the corresponding grid-size-to-D ratio lives in
STEP_DEFAULT_GRID_FACTOR below.
"""

from __future__ import annotations

from math import sqrt

# Grid factor: g = STEP_DEFAULT_GRID_FACTOR * D, i.e. n = 1 in the DASFAA
# formula g = (sqrt(2) / (4 n)) D. This is the authors' baseline choice.
STEP_DEFAULT_GRID_FACTOR: float = sqrt(2) / 4.0
