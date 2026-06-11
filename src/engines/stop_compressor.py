"""Stop-segment centroid compressor (Module II)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

from constants.stop_compression_defaults import (
    StopCompressionStrategy,
    STOP_COMPRESSION_DEFAULT_STRATEGY,
)
from core.compression import BYTES_PER_POINT, SegmentResult
from core.point import Point
from core.segment import Stop


@dataclass(frozen=True)
class CompressedStop:
    centroid: Point  # The representative keypoint; named centroid for compatibility.
    start_time: datetime
    end_time: datetime


class StopCompressor:
    def __init__(
        self,
        strategy: StopCompressionStrategy = STOP_COMPRESSION_DEFAULT_STRATEGY,
        preserve_road_id: bool = False,
    ):
        self.strategy = strategy
        self.preserve_road_id = preserve_road_id

    def compress(self, points: List[Point]) -> CompressedStop:
        """Reduce a stop's points to one representative keypoint and its time range."""
        if not points:
            raise ValueError("Cannot compress empty list of points")

        start_time = points[0].timestamp
        end_time = points[-1].timestamp

        if self.strategy == StopCompressionStrategy.FIRST_POINT:
            keypoint = points[0]

        elif self.strategy == StopCompressionStrategy.MEDOID:
            # Exact medoid minimising summed distance to all other points.
            best_point = None
            min_sum_dist = float('inf')
            for p1 in points:
                sum_dist = 0.0
                for p2 in points:
                    dlat = p1.lat - p2.lat
                    dlon = (p1.lon - p2.lon) * math.cos(math.radians((p1.lat + p2.lat) / 2.0))
                    sum_dist += math.sqrt(dlat*dlat + dlon*dlon)
                if sum_dist < min_sum_dist:
                    min_sum_dist = sum_dist
                    best_point = p1
            keypoint = best_point

        else:
            # CENTROID or SNAP_TO_NEAREST
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            centroid_lat = sum(lats) / len(lats)
            centroid_lon = sum(lons) / len(lons)

            def _dist_sq_to_centroid(p: Point) -> float:
                dlat = p.lat - centroid_lat
                dlon = (p.lon - centroid_lon) * math.cos(math.radians((p.lat + centroid_lat) / 2.0))
                return dlat * dlat + dlon * dlon

            if self.strategy == StopCompressionStrategy.SNAP_TO_NEAREST:
                keypoint = min(points, key=_dist_sq_to_centroid)
            else:
                # CENTROID: synthetic position; snap scan only when road_id is needed.
                keypoint = Point(
                    lat=centroid_lat,
                    lon=centroid_lon,
                    timestamp=start_time,
                    obj_id=points[0].obj_id,
                    road_id=(
                        min(points, key=_dist_sq_to_centroid).road_id
                        if self.preserve_road_id
                        else None
                    ),
                )

        # Normalise the timestamp to start_time; strip road_id unless preserved.
        final_keypoint = Point(
            lat=keypoint.lat,
            lon=keypoint.lon,
            timestamp=start_time,
            obj_id=keypoint.obj_id,
            road_id=keypoint.road_id if self.preserve_road_id else None,
        )

        return CompressedStop(
            centroid=final_keypoint,
            start_time=start_time,
            end_time=end_time
        )

    def compress_segment(self, stop: Stop) -> SegmentResult:
        """Compress a Stop segment to a single-keypoint SegmentResult."""
        cs = self.compress(stop.points)
        return SegmentResult(
            kind="stop",
            start_time=stop.start_time,
            end_time=stop.end_time,
            keypoints=[cs.centroid],
            encoded_bytes=BYTES_PER_POINT,
        )

    @staticmethod
    def passthrough_segment(stop: Stop) -> SegmentResult:
        """Wrap a Stop segment as a SegmentResult without compression."""
        return SegmentResult(
            kind="stop",
            start_time=stop.start_time,
            end_time=stop.end_time,
            keypoints=list(stop.points),
            encoded_bytes=len(stop.points) * BYTES_PER_POINT,
        )
