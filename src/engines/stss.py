"""STSS offline density-based stop/move segmenter (Oracle-G / Oracle-N Module I)."""

from __future__ import annotations

from math import radians
from typing import List

import numpy as np
from sklearn.cluster import OPTICS

from constants.geo_defaults import EARTH_RADIUS_M
from constants.segmentation_defaults import (
    STSS_MAX_EPS_METERS,
    STSS_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from core.point import Point
from core.segment import Move, Segment, Stop


DEFAULT_XI: float = 0.02


class STSSOracle:
    """Offline density-based stop/move segmenter for the Oracle pipelines."""

    def __init__(
        self,
        min_samples: int = STSS_MIN_SAMPLES,
        max_eps: float = STSS_MAX_EPS_METERS,
        min_duration_seconds: float = STSS_MIN_DURATION_SECONDS,
        xi: float = DEFAULT_XI,
    ):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.min_duration_seconds = min_duration_seconds
        self.xi = xi

    def process(self, trajectory: List[Point]) -> List[Segment]:
        if not trajectory:
            return []

        # OPTICS with haversine expects [lat, lon] in radians.
        coords = np.array([[radians(p.lat), radians(p.lon)] for p in trajectory])
        max_eps_rad = self.max_eps / EARTH_RADIUS_M

        clustering = OPTICS(
            min_samples=self.min_samples,
            max_eps=max_eps_rad,
            metric="haversine",
            xi=self.xi,
        )
        clustering.fit(coords)
        labels = clustering.labels_

        return self._post_process(self._labels_to_segments(trajectory, labels))

    @staticmethod
    def _labels_to_segments(
        trajectory: List[Point], labels: np.ndarray
    ) -> List[Segment]:
        """Walk labels in trajectory order, emitting Stop / Move segments."""
        if not trajectory:
            return []

        segments: List[Segment] = []
        current_points: List[Point] = [trajectory[0]]
        current_label = int(labels[0])

        for i in range(1, len(trajectory)):
            label = int(labels[i])
            if label != current_label:
                STSSOracle._flush_segment(segments, current_points, current_label)
                current_points = [trajectory[i]]
                current_label = label
            else:
                current_points.append(trajectory[i])

        STSSOracle._flush_segment(segments, current_points, current_label)
        return segments

    @staticmethod
    def _flush_segment(
        segments: List[Segment], points: List[Point], label: int
    ) -> None:
        if not points:
            return
        if label == -1:
            segments.append(Move(points=points))
            return
        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        centroid = Point(
            lat=sum(lats) / len(lats),
            lon=sum(lons) / len(lons),
            timestamp=points[0].timestamp,
            obj_id=points[0].obj_id,
        )
        segments.append(Stop(points=points, centroid=centroid))

    def _post_process(self, segments: List[Segment]) -> List[Segment]:
        """Demote short Stops to Moves and merge adjacent Moves."""
        if not segments:
            return []

        kept: List[Segment] = []
        for seg in segments:
            if isinstance(seg, Stop):
                duration = (seg.end_time - seg.start_time).total_seconds()
                if duration < self.min_duration_seconds:
                    kept.append(Move(points=seg.points))
                else:
                    kept.append(seg)
            else:
                kept.append(seg)

        merged: List[Segment] = []
        current = kept[0]
        for nxt in kept[1:]:
            if isinstance(current, Move) and isinstance(nxt, Move):
                current = Move(points=current.points + nxt.points)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged
