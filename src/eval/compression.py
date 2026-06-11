"""Compression-ratio metrics."""

from core.compression import TrajectoryResult


def calculate_compression_ratio(result: TrajectoryResult) -> float:
    """Byte-based compression ratio: original_bytes / encoded_bytes."""
    return result.compression_ratio
