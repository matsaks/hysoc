"""Default parameters for TRACE move compression."""

from __future__ import annotations

# Speed threshold (gamma in TRACE).
TRACE_GAMMA: float = 50.0

# Error bound for prediction (epsilon in TRACE).
TRACE_EPSILON: float = 15.0

# k-mer length.
TRACE_K: int = 4

# Threshold for reference rewriting (alpha in TRACE).
TRACE_ALPHA: int = 5

# Threshold C for reference deletion.
TRACE_CLEANUP_THRESHOLD: float = 1000.0

# Decay factor for freshness.
TRACE_DECAY_LAMBDA: float = 0.9

