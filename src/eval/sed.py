"""Synchronized Euclidean Distance (SED) metrics."""

from __future__ import annotations

import math

from core.point import Point
from core.compression import TrajectoryResult


def calculate_sed_error(p_original: Point, p_start: Point, p_end: Point) -> float:
    """SED error (metres) for a point relative to the segment [p_start, p_end]."""
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
    original: list[Point], compressed: list[Point]
) -> dict[str, float | list[float]]:
    """SED statistics comparing original points against a compressed keypoint list."""
    if not original or not compressed:
        return {
            "average_sed": 0.0,
            "p95_sed": 0.0,
            "max_sed": 0.0,
            "rmse": 0.0,
            "sed_errors": [],
        }

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
        return {
            "average_sed": 0.0,
            "p95_sed": 0.0,
            "max_sed": 0.0,
            "rmse": 0.0,
            "sed_errors": [],
        }

    ordered = sorted(sed_errors)
    idx_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    avg_sed = sum(sed_errors) / len(sed_errors)
    max_sed = max(sed_errors)
    rmse = math.sqrt(sum(e * e for e in sed_errors) / len(sed_errors))

    return {
        "average_sed": avg_sed,
        "p95_sed": ordered[idx_95],
        "max_sed": max_sed,
        "rmse": rmse,
        "sed_errors": sed_errors,
    }


def calculate_sed_from_result(result: TrajectoryResult) -> dict[str, float | list[float]]:
    """Convenience wrapper: extracts original_points and keypoints from a TrajectoryResult."""
    return calculate_sed_stats(result.original_points, result.keypoints)


def calculate_move_sed_stats(result: TrajectoryResult) -> dict[str, float]:
    """Per-move SED statistics (mean and max) aggregated across all Move segments."""
    moves = result.moves()
    if not moves:
        return {"mean_sed": 0.0, "max_sed": 0.0, "n_moves": 0}

    orig_by_time = sorted(result.original_points, key=lambda p: p.timestamp)

    move_mean_seds: list[float] = []
    move_max_seds: list[float] = []

    for seg in moves:
        if len(seg.keypoints) < 2:
            continue
        seg_points = [
            p for p in orig_by_time
            if seg.start_time <= p.timestamp <= seg.end_time
        ]
        if not seg_points:
            continue
        stats = calculate_sed_stats(seg_points, seg.keypoints)
        move_mean_seds.append(stats["average_sed"])
        move_max_seds.append(stats["max_sed"])

    if not move_mean_seds:
        return {"mean_sed": 0.0, "max_sed": 0.0, "n_moves": 0}

    return {
        "mean_sed": sum(move_mean_seds) / len(move_mean_seds),
        "max_sed": sum(move_max_seds) / len(move_max_seds),
        "n_moves": len(move_mean_seds),
    }
