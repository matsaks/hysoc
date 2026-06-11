"""Configuration for the hybrid SQUISH + Douglas-Peucker move compressor."""

from dataclasses import dataclass

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY


@dataclass(frozen=True)
class HybridSquishDPConfig:
    """Configuration for Hybrid SQUISH + DP move compression (HYSOC-G)."""

    capacity: int = SQUISH_DEFAULT_CAPACITY
    dp_epsilon_meters: float = DP_DEFAULT_EPSILON_METERS
    dp_refine_when_evictions: bool = True
