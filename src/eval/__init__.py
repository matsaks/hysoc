"""Evaluation metrics for compression, SED, segmentation, and latency."""

from .compression import calculate_compression_ratio
from .latency import (
    LATENCY_DEFAULT_REPEATS,
    LATENCY_DEFAULT_WARMUP,
    LatencyStats,
    calculate_latency_stats,
    measure_latency,
)
from .sed import calculate_sed_error, calculate_sed_stats, calculate_sed_from_result
from .segmentation import (
    F1Result,
    segment_counts,
    segment_counts_from_result,
    stop_temporal_iou,
    stop_f1,
    stop_f1_from_result,
    road_segment_jaccard,
    road_segment_jaccard_vs_original,
)

__all__ = [
    "calculate_compression_ratio",
    "LATENCY_DEFAULT_REPEATS",
    "LATENCY_DEFAULT_WARMUP",
    "LatencyStats",
    "calculate_latency_stats",
    "measure_latency",
    "calculate_sed_error",
    "calculate_sed_stats",
    "calculate_sed_from_result",
    "F1Result",
    "segment_counts",
    "segment_counts_from_result",
    "stop_temporal_iou",
    "stop_f1",
    "stop_f1_from_result",
    "road_segment_jaccard",
    "road_segment_jaccard_vs_original",
]
