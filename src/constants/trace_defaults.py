"""TRACE move-compression defaults."""

from __future__ import annotations

# Speed-change threshold for retaining a point (m/s).
TRACE_GAMMA: float = 5.0

# Speed quantisation bin width.
TRACE_EPSILON: float = 15.0

# k-mer length for referential matching against the shared reference index.
TRACE_K: int = 5

# Reference-pruning freshness fraction, in (0, 1).
TRACE_CLEANUP_THRESHOLD: float = 0.5

# Freshness decay factor.
TRACE_DECAY_LAMBDA: float = 0.9
