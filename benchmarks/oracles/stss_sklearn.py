from typing import List
import numpy as np
from sklearn.cluster import OPTICS
from math import radians

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move

class STSSOracleSklearn:
    """
    Implements the STSS (Semantics-Based Trajectory Segmentation Simplification) method
    using Scikit-Learn's OPTICS implementation.
    Reference: Liu et al. (2021)
    """

    def __init__(self, min_samples: int = 2, max_eps: float = 50.0, min_duration_seconds: float = 60.0):
        """
        Args:
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            max_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. (Meters)
            min_duration_seconds: Minimum duration for a cluster to be considered a Stop.
        """
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.min_duration_seconds = min_duration_seconds

    def process(self, trajectory: List[Point]) -> List[Segment]:
        if not trajectory:
            return []

        # 1. Prepare data for OPTICS
        # Scikit-learn haversine metric expects [lat, lon] in RADIANS
        coords = np.array([[radians(p.lat), radians(p.lon)] for p in trajectory])
        
        # Earth radius in meters approximately 6371000
        # max_eps is in meters, convert to radians: dist / radius
        max_eps_rad = self.max_eps / 6371000.0

        clustering = OPTICS(min_samples=self.min_samples, max_eps=max_eps_rad, metric='haversine')
        clustering.fit(coords)
        
        labels = clustering.labels_

        # 2. Extract Segments based on labels
        # -1 is noise (Move)
        # 0, 1, ... are clusters (Potential Stops)
        
        segments: List[Segment] = []
        current_points: List[Point] = []
        current_label = labels[0] if len(labels) > 0 else -1

        for i, point in enumerate(trajectory):
            label = labels[i]

            if label != current_label:
                # Flush current segment
                self._add_segment(segments, current_points, current_label)
                current_points = [point]
                current_label = label
            else:
                current_points.append(point)
        
        # Flush last
        self._add_segment(segments, current_points, current_label)

        # 3. Post-process: Filter short stops and merge adjacent moves
        return self._post_process(segments)

    def _add_segment(self, segments: List[Segment], points: List[Point], label: int):
        if not points:
            return
            
        if label == -1:
            segments.append(Move(points=points))
        else:
            # OPTICS cluster -> Candidate Stop
            # Centroid calculation
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            centroid = Point(
                lat=sum(lats) / len(lats),
                lon=sum(lons) / len(lons),
                timestamp=points[0].timestamp, # Use start time as reference
                obj_id=points[0].obj_id
            )
            segments.append(Stop(points=points, centroid=centroid))

    def _post_process(self, segments: List[Segment]) -> List[Segment]:
        """
        Filters out Stops that are too short (convert to Move) and merges adjacent Moves.
        """
        # 1. Convert short Stops to Moves
        converted_segments = []
        for seg in segments:
            if isinstance(seg, Stop):
                duration = (seg.end_time - seg.start_time).total_seconds()
                if duration < self.min_duration_seconds:
                    converted_segments.append(Move(points=seg.points))
                else:
                    converted_segments.append(seg)
            else:
                converted_segments.append(seg)
        
        # 2. Merge adjacent Moves
        if not converted_segments:
            return []

        merged_segments = []
        current_seg = converted_segments[0]

        for next_seg in converted_segments[1:]:
            if isinstance(current_seg, Move) and isinstance(next_seg, Move):
                # Merge
                # Note: This simple merge concatenates points. 
                # Ideally check for duplicates at boundaries if they share points, 
                # but stream/pipeline usually produces distinct points.
                current_seg = Move(points=current_seg.points + next_seg.points)
            else:
                merged_segments.append(current_seg)
                current_seg = next_seg
        
        merged_segments.append(current_seg)
        return merged_segments
