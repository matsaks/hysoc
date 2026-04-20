"""
Segmentation-quality metrics for Stop/Move segmentation.

The primary matching rule is **temporal IoU (Jaccard)** as defined in the
TDT4501 preproject (Sect. 4.6.1, Eq. 6):

    duration(S_det ∩ S_gt) / duration(S_det ∪ S_gt) >= 0.5

This is stricter than the preproject's prose ("overlaps ... by at least
50 %"), which is an informal gloss. Implementation follows Eq. 6 literally.

The module exposes three helpers:

- `segment_counts`: descriptive counts used by the parameter sweep.
- `stop_temporal_iou`: the pairwise matching function.
- `stop_f1`: Precision / Recall / F1 under the IoU >= 0.5 rule, used by
  RQ1 experiments (preproject Sect. 4.6.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from core.segment import Segment, Stop, Move


def segment_counts(segments: Iterable[Segment]) -> dict[str, float]:
    """
    Compute descriptive counts for a list of Stop/Move segments.

    Returns a dict with keys:
      - n_stops, n_moves
      - mean_stop_duration_s, mean_move_duration_s
      - mean_stop_points, mean_move_points
      - total_points

    Missing categories yield 0 counts and 0.0 means (no NaN propagation).
    """
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

    total_points = sum(len(s.points) for s in stops) + sum(
        len(s.points) for s in moves
    )

    return {
        "n_stops": len(stops),
        "n_moves": len(moves),
        "mean_stop_duration_s": _mean_duration(stops),
        "mean_move_duration_s": _mean_duration(moves),
        "mean_stop_points": _mean_points(stops),
        "mean_move_points": _mean_points(moves),
        "total_points": total_points,
    }


def stop_temporal_iou(a: Stop, b: Stop) -> float:
    """
    Temporal IoU (Jaccard) between two Stop segments.

    Returns `duration(a ∩ b) / duration(a ∪ b)` in the closed time domain.
    Zero-duration stops (single-point) are allowed: intersection is 0 unless
    the single instant is contained in the other segment, in which case
    Jaccard reduces to 0 / |b| = 0.
    """
    if not a.points or not b.points:
        return 0.0

    a_start, a_end = a.start_time, a.end_time
    b_start, b_end = b.start_time, b.end_time

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
    """
    F1 score for stop detection using temporal IoU matching.

    A predicted Stop is a True Positive iff it has IoU >= threshold with
    some unmatched ground-truth Stop (greedy 1:1 matching by best IoU,
    iterating predictions in chronological order).

    Args:
        predicted: segments produced by the algorithm under test (Stops and
            Moves are accepted; Moves are ignored).
        ground_truth: segments produced by the reference oracle.
        temporal_iou_threshold: minimum IoU for a match. Default 0.5 per
            TDT4501 preproject Eq. 6.

    Returns:
        F1Result with precision, recall, f1, TP/FP/FN counts, and the mean
        IoU over matched pairs (0.0 if no matches).
    """
    pred_stops = sorted(
        (s for s in predicted if isinstance(s, Stop) and s.points),
        key=lambda s: s.start_time,
    )
    gt_stops = sorted(
        (s for s in ground_truth if isinstance(s, Stop) and s.points),
        key=lambda s: s.start_time,
    )

    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    tp = 0

    for p in pred_stops:
        best_iou = 0.0
        best_idx = -1
        for i, g in enumerate(gt_stops):
            if i in matched_gt:
                continue
            iou = stop_temporal_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= temporal_iou_threshold:
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
            tp += 1

    fp = len(pred_stops) - tp
    fn = len(gt_stops) - tp

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
