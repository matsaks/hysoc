from dataclasses import dataclass

from constants.trace_defaults import (
    TRACE_ALPHA,
    TRACE_CLEANUP_THRESHOLD,
    TRACE_DECAY_LAMBDA,
    TRACE_EPSILON,
    TRACE_GAMMA,
    TRACE_K,
)


@dataclass
class TraceConfig:
    """Configuration for TRACE compressor."""

    gamma: float = TRACE_GAMMA
    epsilon: float = TRACE_EPSILON
    k: int = TRACE_K
    alpha: int = TRACE_ALPHA
    cleanup_threshold: float = TRACE_CLEANUP_THRESHOLD
    decay_lambda: float = TRACE_DECAY_LAMBDA
