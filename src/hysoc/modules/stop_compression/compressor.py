from dataclasses import dataclass
from datetime import datetime
from typing import List
from hysoc.core.point import Point

@dataclass(frozen=True)
class CompressedStop:
    centroid: Point
    start_time: datetime
    end_time: datetime

class StopCompressor:
    def compress(self, points: List[Point]) -> CompressedStop:
        """
        Compresses a list of points into a single centroid point.
        Returns a CompressedStop containing the centroid and the time range.
        """
        if not points:
            raise ValueError("Cannot compress empty list of points")

        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        
        # Calculate centroid
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
        
        # Determine start and end times
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        
        centroid = Point(
            lat=centroid_lat,
            lon=centroid_lon,
            timestamp=start_time, # Keep start time in centroid as well
            obj_id=points[0].obj_id
        )
        
        return CompressedStop(
            centroid=centroid,
            start_time=start_time,
            end_time=end_time
        )
