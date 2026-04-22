"""Synchronized Euclidean Distance (SED) metrics."""
from __future__ import annotations

import math
from typing import Dict, List

from core.point import Point
from core.compression import TrajectoryResult


def calculate_sed_error(p_original: Point, p_start: Point, p_end: Point) -> float:
    """
    SED error for a single point relative to the segment [p_start, p_end].

    Returns the Euclidean distance (metres) between p_original and its
    temporal projection onto the segment.
    """
    t_orig = p_original.timestamp.timestamp()
    t_start = p_start.timestamp.timestamp()
    t_end = p_end.timestamp.timestamp()

    if t_start == t_end:
        d_lat = p_original.lat - p_start.lat
        d_lon = p_original.lon - p_start.lon
        avg_lat = math.radians((p_original.lat + p_start.lat) / 2.0)
        d_lat_m = d_lat * 111320.0
        d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
        return math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m)

    ratio = (t_orig - t_start) / (t_end - t_start)
    pred_lat = p_start.lat + (p_end.lat - p_start.lat) * ratio
    pred_lon = p_start.lon + (p_end.lon - p_start.lon) * ratio

    d_lat = p_original.lat - pred_lat
    d_lon = p_original.lon - pred_lon
    avg_lat = math.radians((p_start.lat + p_end.lat) / 2.0)
    d_lat_m = d_lat * 111320.0
    d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)

    return math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m)


def calculate_sed_stats(
    original: List[Point], compressed: List[Point]
) -> Dict[str, float | List[float]]:
    """
    SED statistics comparing original points against a compressed keypoint list.

    Returns average_sed, max_sed, rmse, and the full sed_errors list.
    """
    if not original or not compressed:
        return {"average_sed": 0.0, "max_sed": 0.0, "rmse": 0.0, "sed_errors": []}

    sed_errors: list[float] = []
    comp_idx = 0

    for p in original:
        while comp_idx < len(compressed) - 1 and p.timestamp > compressed[comp_idx + 1].timestamp:
            comp_idx += 1

        if comp_idx >= len(compressed) - 1:
            p_ref = compressed[-1]
            d_lat = p.lat - p_ref.lat
            d_lon = p.lon - p_ref.lon
            avg_lat = math.radians((p.lat + p_ref.lat) / 2.0)
            d_lat_m = d_lat * 111320.0
            d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
            sed_errors.append(math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m))
            continue

        p_start = compressed[comp_idx]
        p_end = compressed[comp_idx + 1]

        if p.timestamp < p_start.timestamp:
            d_lat = p.lat - p_start.lat
            d_lon = p.lon - p_start.lon
            avg_lat = math.radians((p.lat + p_start.lat) / 2.0)
            d_lat_m = d_lat * 111320.0
            d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
            sed_errors.append(math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m))
            continue

        sed_errors.append(calculate_sed_error(p, p_start, p_end))

    if not sed_errors:
        return {"average_sed": 0.0, "max_sed": 0.0, "rmse": 0.0, "sed_errors": []}

    avg_sed = sum(sed_errors) / len(sed_errors)
    max_sed = max(sed_errors)
    rmse = math.sqrt(sum(e * e for e in sed_errors) / len(sed_errors))

    return {"average_sed": avg_sed, "max_sed": max_sed, "rmse": rmse, "sed_errors": sed_errors}


def calculate_sed_from_result(result: TrajectoryResult) -> Dict[str, float | List[float]]:
    """Convenience wrapper that extracts original and keypoints from a TrajectoryResult."""
    return calculate_sed_stats(result.original_points, result.keypoints)
