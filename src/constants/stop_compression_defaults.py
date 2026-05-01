"""
Default configuration for stop compression strategies.
"""

from __future__ import annotations
from enum import Enum

class StopCompressionStrategy(Enum):
    """Available strategies for compressing a stop segment into a single representative point."""
    CENTROID = "centroid"
    MEDOID = "medoid"
    SNAP_TO_NEAREST = "snap_to_nearest"
    FIRST_POINT = "first_point"

STOP_COMPRESSION_DEFAULT_STRATEGY: StopCompressionStrategy = StopCompressionStrategy.CENTROID
