"""Compression-ratio metrics."""
from core.compression import TrajectoryResult


def calculate_compression_ratio(result: TrajectoryResult) -> float:
    """
    Byte-based compression ratio: original_bytes / encoded_bytes.

    For point-list strategies (HYSOC-G, oracles) this equals the point-count
    ratio. For TRACE-based strategies (HYSOC-N) it reflects the actual
    encoding cost rather than the residual point count.
    """
    return result.compression_ratio
