"""Stop-compression strategy defaults."""

from __future__ import annotations

from enum import Enum


class StopCompressionStrategy(Enum):
    """Strategy for reducing a stop segment to one representative point."""

    CENTROID = "centroid"
    MEDOID = "medoid"
    SNAP_TO_NEAREST = "snap_to_nearest"
    FIRST_POINT = "first_point"


STOP_COMPRESSION_DEFAULT_STRATEGY: StopCompressionStrategy = StopCompressionStrategy.CENTROID
