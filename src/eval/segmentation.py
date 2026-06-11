"""Segmentation-quality metrics for Stop/Move segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from core.point import Point
from core.segment import Segment, Stop, Move
from core.compression import SegmentResult, TrajectoryResult


def _temporal_iou(
    a_start: datetime, a_end: datetime,
    b_start: datetime, b_end: datetime,
) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = (inter_end - inter_start).total_seconds()
    if inter <= 0.0:
        return 0.0
    a_dur = (a_end - a_start).total_seconds()
    b_dur = (b_end - b_start).total_seconds()
    union = a_dur + b_dur - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def segment_counts(segments: Iterable[Segment]) -> dict[str, float]:
    """Descriptive counts for a list of Stop/Move segments."""
    stops: list[Stop] = []
    moves: list[Move] = []
    for seg in segments:
        if isinstance(seg, Stop):
            stops.append(seg)
        elif isinstance(seg, Move):
            moves.append(seg)

    def _mean_duration(segs: Sequence[Segment]) -> float:
        if not segs:
            return 0.0
        total = 0.0
        for s in segs:
            if not s.points:
                continue
            total += (s.end_time - s.start_time).total_seconds()
        return total / len(segs)

    def _mean_points(segs: Sequence[Segment]) -> float:
        if not segs:
            return 0.0
        return sum(len(s.points) for s in segs) / len(segs)

    total_points = sum(len(s.points) for s in stops) + sum(len(s.points) for s in moves)

    return {
        "n_stops": len(stops),
        "n_moves": len(moves),
        "mean_stop_duration_s": _mean_duration(stops),
        "mean_move_duration_s": _mean_duration(moves),
        "mean_stop_points": _mean_points(stops),
        "mean_move_points": _mean_points(moves),
        "total_points": total_points,
    }


def segment_counts_from_result(result: TrajectoryResult) -> dict[str, float]:
    """Descriptive counts over a TrajectoryResult."""
    stops = result.stops()
    moves = result.moves()

    def _mean_duration(segs: list[SegmentResult]) -> float:
        if not segs:
            return 0.0
        return sum((s.end_time - s.start_time).total_seconds() for s in segs) / len(segs)

    def _mean_keypoints(segs: list[SegmentResult]) -> float:
        if not segs:
            return 0.0
        return sum(len(s.keypoints) for s in segs) / len(segs)

    return {
        "n_stops": len(stops),
        "n_moves": len(moves),
        "mean_stop_duration_s": _mean_duration(stops),
        "mean_move_duration_s": _mean_duration(moves),
        "mean_stop_keypoints": _mean_keypoints(stops),
        "mean_move_keypoints": _mean_keypoints(moves),
        "total_keypoints": float(len(result.keypoints)),
    }


def stop_temporal_iou(a: Stop, b: Stop) -> float:
    """Temporal IoU (Jaccard) between two Stop segments."""
    if not a.points or not b.points:
        return 0.0
    return _temporal_iou(a.start_time, a.end_time, b.start_time, b.end_time)


@dataclass(frozen=True)
class F1Result:
    """Precision / Recall / F1 with supporting counts."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_iou_mean: float


def stop_f1(
    predicted: Iterable[Segment],
    ground_truth: Iterable[Segment],
    temporal_iou_threshold: float = 0.5,
) -> F1Result:
    """F1 score for stop detection over raw Stop/Move segments."""
    pred_stops = sorted(
        (s for s in predicted if isinstance(s, Stop) and s.points),
        key=lambda s: s.start_time,
    )
    gt_stops = sorted(
        (s for s in ground_truth if isinstance(s, Stop) and s.points),
        key=lambda s: s.start_time,
    )
    return _f1_from_stop_lists(
        [(s.start_time, s.end_time) for s in pred_stops],
        [(s.start_time, s.end_time) for s in gt_stops],
        temporal_iou_threshold,
    )


def stop_f1_from_result(
    predicted: TrajectoryResult,
    ground_truth: TrajectoryResult,
    temporal_iou_threshold: float = 0.5,
) -> F1Result:
    """F1 score for stop detection between two TrajectoryResults."""
    pred_stops = sorted(predicted.stops(), key=lambda s: s.start_time)
    gt_stops = sorted(ground_truth.stops(), key=lambda s: s.start_time)
    return _f1_from_stop_lists(
        [(s.start_time, s.end_time) for s in pred_stops],
        [(s.start_time, s.end_time) for s in gt_stops],
        temporal_iou_threshold,
    )


def road_segment_jaccard(
    predicted: TrajectoryResult,
    ground_truth: TrajectoryResult,
) -> float:
    """Jaccard similarity between the road-segment sets of two trajectories."""
    pred_ids = {
        p.road_id
        for seg in predicted.segments
        for p in seg.keypoints
        if p.road_id is not None
    }
    gt_ids = {
        p.road_id
        for seg in ground_truth.segments
        for p in seg.keypoints
        if p.road_id is not None
    }

    if not pred_ids and not gt_ids:
        return float("nan")
    if not pred_ids or not gt_ids:
        return 0.0

    return len(pred_ids & gt_ids) / len(pred_ids | gt_ids)


def road_segment_jaccard_vs_original(
    result: TrajectoryResult,
    original_points: list[Point],
) -> float:
    """Road-segment Jaccard between a compressed result and the original GPS trace."""
    original_ids = {p.road_id for p in original_points if p.road_id is not None}
    compressed_ids = {
        p.road_id
        for seg in result.segments
        for p in seg.keypoints
        if p.road_id is not None
    }
    if not original_ids and not compressed_ids:
        return float("nan")
    if not original_ids or not compressed_ids:
        return 0.0
    return len(original_ids & compressed_ids) / len(original_ids | compressed_ids)


def _f1_from_stop_lists(
    predicted: list[tuple[datetime, datetime]],
    ground_truth: list[tuple[datetime, datetime]],
    threshold: float,
) -> F1Result:
    # No predicted or ground-truth stops: metric is undefined (nan), not 0.0,
    # so the trajectory drops out of population medians.
    if not predicted and not ground_truth:
        nan = float("nan")
        return F1Result(
            precision=nan,
            recall=nan,
            f1=nan,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            matched_iou_mean=nan,
        )

    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    tp = 0

    for p_start, p_end in predicted:
        best_iou = 0.0
        best_idx = -1
        for i, (g_start, g_end) in enumerate(ground_truth):
            if i in matched_gt:
                continue
            iou = _temporal_iou(p_start, p_end, g_start, g_end)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= threshold:
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
            tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return F1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        matched_iou_mean=mean_iou,
    )
