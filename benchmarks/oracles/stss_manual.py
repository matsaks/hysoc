from typing import List, Set
from math import radians, sin, cos, sqrt, atan2

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move

def haversine_distance(p1: Point, p2: Point) -> float:
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    Returns distance in meters.
    """
    # Earth radius in meters
    R = 6371000.0

    lat1, lon1 = radians(p1.lat), radians(p1.lon)
    lat2, lon2 = radians(p2.lat), radians(p2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

class STSSOracleManual:
    """
    A manual implementation of density-based trajectory segmentation (STSS).
    Uses a DBSCAN-like approach to identify clusters of points (potential Stops)
    based on spatial density, without external ML dependencies.
    """

    def __init__(self, min_samples: int = 2, max_eps: float = 50.0, min_duration_seconds: float = 60.0):
        """
        Args:
            min_samples: Minimum neighbors to form a core point.
            max_eps: Maximum distance (meters) to consider a neighbor.
            min_duration_seconds: Minimum duration to keep a Stop.
        """
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.min_duration_seconds = min_duration_seconds

    def process(self, trajectory: List[Point]) -> List[Segment]:
        if not trajectory:
            return []

        n = len(trajectory)
        labels = [-1] * n  # -1 represents Noise (Move)
        cluster_id = 0

        # Precompute distances is O(N^2), might be heavy for large N.
        # Check size. If N < 5000, O(N^2) is ~25M ops, manageable in python for offline oracle.
        # For larger, would need spatial index. Assuming offline oracle accepts some slowness.
        
        # We will do on-demand distance calculation to save memory, 
        # but still O(N^2) worst case.
        
        visited = [False] * n

        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            
            neighbors = self._region_query(trajectory, i)
            
            if len(neighbors) < self.min_samples:
                # Mark as Noise (-1) - already default
                continue
            else:
                # Start new cluster
                cluster_id += 0  # 0-indexed labels for consistency? 
                # Actually let's use 0, 1, 2...
                # Current cluster_id is 0
                
                labels[i] = cluster_id
                
                # Expand cluster
                # neighbors is list of indices
                seeds = list(neighbors)
                # Remove i from seeds if present (region_query usually includes self? DBSCAN logic typically does)
                # We'll just iterate over seeds
                
                while seeds:
                    curr_idx = seeds.pop()
                    
                    if not visited[curr_idx]:
                        visited[curr_idx] = True
                        curr_neighbors = self._region_query(trajectory, curr_idx)
                        
                        if len(curr_neighbors) >= self.min_samples:
                            seeds.extend(curr_neighbors)
                    
                    if labels[curr_idx] == -1:
                        labels[curr_idx] = cluster_id

                cluster_id += 1

        # Now we have labels. Convert to Segments.
        # This part is identical to the Sklearn version, we can duplicate logic or refactor (but creating standalone file is requested)
        return self._labels_to_segments(trajectory, labels)

    def _region_query(self, trajectory: List[Point], center_idx: int) -> List[int]:
        neighbors = []
        center_p = trajectory[center_idx]
        for i, p in enumerate(trajectory):
            dist = haversine_distance(center_p, p)
            if dist <= self.max_eps:
                neighbors.append(i)
        return neighbors

    def _labels_to_segments(self, trajectory: List[Point], labels: List[int]) -> List[Segment]:
        segments: List[Segment] = []
        current_points: List[Point] = []
        current_label = labels[0] if len(labels) > 0 else -1

        for i, point in enumerate(trajectory):
            label = labels[i]

            if label != current_label:
                self._add_segment(segments, current_points, current_label)
                current_points = [point]
                current_label = label
            else:
                current_points.append(point)
        
        self._add_segment(segments, current_points, current_label)

        # Post-process
        return self._post_process(segments)

    def _add_segment(self, segments: List[Segment], points: List[Point], label: int):
        if not points:
            return
            
        if label == -1:
            segments.append(Move(points=points))
        else:
            # Stop
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            centroid = Point(
                lat=sum(lats) / len(lats),
                lon=sum(lons) / len(lons),
                timestamp=points[0].timestamp,
                obj_id=points[0].obj_id
            )
            segments.append(Stop(points=points, centroid=centroid))

    def _post_process(self, segments: List[Segment]) -> List[Segment]:
        # 1. Check duration
        converted = []
        for seg in segments:
            if isinstance(seg, Stop):
                duration = (seg.end_time - seg.start_time).total_seconds()
                if duration < self.min_duration_seconds:
                    converted.append(Move(points=seg.points))
                else:
                    converted.append(seg)
            else:
                converted.append(seg)
        
        # 2. Merge adjacent Moves
        if not converted:
            return []
            
        merged = []
        current = converted[0]
        
        for next_seg in converted[1:]:
            if isinstance(current, Move) and isinstance(next_seg, Move):
                current = Move(points=current.points + next_seg.points)
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        return merged
