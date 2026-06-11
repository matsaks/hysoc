"""Per-city operating points bundling every calibrated pipeline parameter."""

from __future__ import annotations

from dataclasses import dataclass

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from constants.trace_defaults import (
    TRACE_CLEANUP_THRESHOLD,
    TRACE_DECAY_LAMBDA,
    TRACE_EPSILON,
    TRACE_GAMMA,
    TRACE_K,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Frozen bundle of the parameters selected by the calibration sweeps."""

    name: str
    calibrated_on: str

    # Module I (STEP / STSS).
    stop_max_eps_m: float
    stop_min_duration_s: float
    stss_min_samples: int

    # Oracle-G + HYSOC-G Module III.
    dp_epsilon_m: float
    squish_capacity: int

    # HYSOC-N Module III (TRACE).
    trace_gamma: float
    trace_epsilon: float
    trace_k: int
    trace_cleanup_threshold: float
    trace_decay_lambda: float


NYC_OPERATING_POINT = OperatingPoint(
    name="nyc",
    calibrated_on="NYC_Calibration_100",
    stop_max_eps_m=STOP_MAX_EPS_METERS,
    stop_min_duration_s=STOP_MIN_DURATION_SECONDS,
    stss_min_samples=STSS_MIN_SAMPLES,
    dp_epsilon_m=DP_DEFAULT_EPSILON_METERS,
    squish_capacity=SQUISH_DEFAULT_CAPACITY,
    trace_gamma=TRACE_GAMMA,
    trace_epsilon=TRACE_EPSILON,
    trace_k=TRACE_K,
    trace_cleanup_threshold=TRACE_CLEANUP_THRESHOLD,
    trace_decay_lambda=TRACE_DECAY_LAMBDA,
)

# squish_capacity inherits the NYC value; no SF configuration met the SED budget.
SF_OPERATING_POINT = OperatingPoint(
    name="sf_centre",
    calibrated_on="SanFranCentre_Calibration_100",
    stop_max_eps_m=10.0,
    stop_min_duration_s=15.0,
    stss_min_samples=max(5, round(15.0 * 0.5)),
    dp_epsilon_m=1.0,
    squish_capacity=SQUISH_DEFAULT_CAPACITY,
    trace_gamma=5.0,
    trace_epsilon=TRACE_EPSILON,
    trace_k=5,
    trace_cleanup_threshold=TRACE_CLEANUP_THRESHOLD,
    trace_decay_lambda=TRACE_DECAY_LAMBDA,
)

OPERATING_POINTS: dict[str, OperatingPoint] = {
    "nyc": NYC_OPERATING_POINT,
    "sf": SF_OPERATING_POINT,
}

DEFAULT_OPERATING_POINT: str = "nyc"
