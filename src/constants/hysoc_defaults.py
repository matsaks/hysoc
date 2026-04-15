"""
Default parameters for the HYSOC unified compression pipeline.

These constants control the orchestrator behaviour and are shared
across scripts, demos, and benchmarks.
"""

from __future__ import annotations

# Default move-compression strategy identifier.
HYSOC_DEFAULT_STRATEGY: str = "geometric"  # "geometric" (HYSOC-G) or "network_semantic" (HYSOC-N)

# Whether stop segments should be compressed to a single centroid by default.
HYSOC_DEFAULT_COMPRESS_STOPS: bool = True
