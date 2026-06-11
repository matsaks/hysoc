"""Per-element latency measurement and aggregation utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Callable, Iterable


LATENCY_DEFAULT_WARMUP: int = 1
LATENCY_DEFAULT_REPEATS: int = 5


@dataclass(frozen=True)
class LatencyStats:
    """Per-element latency statistics in microseconds."""

    n_elements: int
    warmup: int
    n_repeats: int
    median_us: float
    p25_us: float
    p75_us: float
    mean_us: float
    std_us: float
    min_us: float
    max_us: float


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of a sorted list (q in [0, 1])."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def measure_latency(
    work: Callable[[], object],
    n_elements: int,
    *,
    warmup: int = LATENCY_DEFAULT_WARMUP,
    repeats: int = LATENCY_DEFAULT_REPEATS,
) -> LatencyStats:
    """Time ``work`` over warmup + repeats runs and return per-element latency statistics."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if n_elements <= 0:
        nan = float("nan")
        return LatencyStats(
            n_elements=n_elements,
            warmup=warmup,
            n_repeats=repeats,
            median_us=nan,
            p25_us=nan,
            p75_us=nan,
            mean_us=nan,
            std_us=nan,
            min_us=nan,
            max_us=nan,
        )

    for _ in range(max(0, warmup)):
        work()

    per_element_us: list[float] = []
    for _ in range(repeats):
        t_start = time.perf_counter_ns()
        work()
        elapsed_ns = time.perf_counter_ns() - t_start
        per_element_us.append(elapsed_ns / 1000.0 / n_elements)

    sorted_us = sorted(per_element_us)
    return LatencyStats(
        n_elements=n_elements,
        warmup=warmup,
        n_repeats=repeats,
        median_us=median(sorted_us),
        p25_us=_percentile(sorted_us, 0.25),
        p75_us=_percentile(sorted_us, 0.75),
        mean_us=mean(sorted_us),
        std_us=stdev(sorted_us) if len(sorted_us) > 1 else 0.0,
        min_us=sorted_us[0],
        max_us=sorted_us[-1],
    )


def calculate_latency_stats(latencies_seconds: Iterable[float]) -> dict[str, float]:
    """Aggregate pre-recorded per-point/per-stage latency values into summary stats."""
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
