from __future__ import annotations

from statistics import mean
from typing import Iterable


def calculate_latency_stats(latencies_seconds: Iterable[float]) -> dict[str, float]:
    """Return simple latency stats for per-point/per-stage timing values."""
    values = [float(v) for v in latencies_seconds]
    if not values:
        return {"count": 0.0, "mean_s": 0.0, "max_s": 0.0, "min_s": 0.0, "p95_s": 0.0}

    ordered = sorted(values)
    idx_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": float(len(values)),
        "mean_s": float(mean(values)),
        "max_s": float(max(values)),
        "min_s": float(min(values)),
        "p95_s": float(ordered[idx_95]),
    }


def calculate_pipeline_latency_from_diagnostics(diagnostics: dict) -> dict[str, float]:
    """Aggregate total stage latency from compressor diagnostics."""
    keys = (
        "map_matching_time_s",
        "segmentation_time_s",
        "compression_time_s",
        "trace_time_s",
        "trace_retained_extract_time_s",
    )
    total = sum(float(diagnostics.get(k, 0.0)) for k in keys)
    return {"total_pipeline_time_s": total}
