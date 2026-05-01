import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

from core.point import Point
from constants.stop_compression_defaults import StopCompressionStrategy, STOP_COMPRESSION_DEFAULT_STRATEGY

@dataclass(frozen=True)
class CompressedStop:
    centroid: Point  # Note: Kept name as 'centroid' for backward compatibility, but it represents the 'keypoint'
    start_time: datetime
    end_time: datetime

class StopCompressor:
    def __init__(self, strategy: StopCompressionStrategy = STOP_COMPRESSION_DEFAULT_STRATEGY):
        self.strategy = strategy

    def compress(self, points: List[Point]) -> CompressedStop:
        """
        Compresses a list of points into a single point using the configured strategy.
        Returns a CompressedStop containing the representative point and the time range.
        """
        if not points:
            raise ValueError("Cannot compress empty list of points")

        start_time = points[0].timestamp
        end_time = points[-1].timestamp

        if self.strategy == StopCompressionStrategy.FIRST_POINT:
            keypoint = points[0]
            
        elif self.strategy == StopCompressionStrategy.MEDOID:
            # O(n^2) exact medoid minimizing sum of distances to all other points
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
            
            centroid_p = Point(
                lat=centroid_lat,
                lon=centroid_lon,
                timestamp=start_time,
                obj_id=points[0].obj_id
            )
            
            if self.strategy == StopCompressionStrategy.SNAP_TO_NEAREST:
                # O(n) find closest raw point to centroid
                best_point = None
                min_dist_sq = float('inf')
                for p in points:
                    dlat = p.lat - centroid_lat
                    dlon = (p.lon - centroid_lon) * math.cos(math.radians((p.lat + centroid_lat) / 2.0))
                    dist_sq = dlat*dlat + dlon*dlon
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_point = p
                keypoint = best_point
            else:
                keypoint = centroid_p

        # Return a CompressedStop, making sure the keypoint timestamp is the start_time 
        # (for consistency with how the centroid behaved previously, though medoid/snap have their own timestamps)
        final_keypoint = Point(
            lat=keypoint.lat,
            lon=keypoint.lon,
            timestamp=start_time,
            obj_id=keypoint.obj_id
        )

        return CompressedStop(
            centroid=final_keypoint,
            start_time=start_time,
            end_time=end_time
        )
