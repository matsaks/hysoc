import math
from typing import List, Tuple, Optional
from datetime import datetime

from hysoc.core.point import Point
from hysoc.core.segment import Segment, Stop, Move

def local_distance(p1: Point, p2: Point) -> float:
    """
    Fast flat-earth distance approximation in meters.
    Adequate for short spatial extents (like detecting local stay-points).
    """
    R = 6371000.0
    lat_rad = math.radians((p1.lat + p2.lat) / 2.0)
    dx = math.radians(p2.lon - p1.lon) * math.cos(lat_rad)
    dy = math.radians(p2.lat - p1.lat)
    return R * math.sqrt(dx*dx + dy*dy)

class STEPSegmenter:
    """
    Streaming Trajectory Segmentation Based on Stay-Point Detection (STEP).
    Identifies STOP and MOVE segments on the fly.
    """

    def __init__(self, max_eps: float = 50.0, min_duration_seconds: float = 60.0, grid_size_meters: Optional[float] = None):
        """
        Args:
            max_eps: Distance threshold D in meters.
            min_duration_seconds: Time threshold T in seconds.
            grid_size_meters: Custom grid cell dimension g. If None, uses default g = (sqrt(2)/4) * D.
        """
        self.max_eps = max_eps
        self.min_duration_seconds = min_duration_seconds
        
        if grid_size_meters is not None and grid_size_meters > 0:
            self.g = grid_size_meters
        else:
            self.g = (math.sqrt(2) / 4.0) * max_eps
            
        self.threshold_sq = (self.max_eps / self.g) ** 2
            
        # Origin for local 2D projection
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        
        # State Arrays
        # cache holds points that are not yet flushed
        # Elements are tuple: (Point, gx, gy)
        self.cache: List[Tuple[Point, int, int]] = []
        self.cache_offset = 0 # The absolute index matching cache[0]
        
        # Absolute indices of currently identified stay-point
        self.current_sp_start: Optional[int] = None
        self.current_sp_end: Optional[int] = None

    def _get_cached_item(self, abs_index: int) -> Tuple[Point, int, int]:
        return self.cache[abs_index - self.cache_offset]
        
    def _get_point(self, abs_index: int) -> Point:
        return self.cache[abs_index - self.cache_offset][0]
        
    def _get_points(self, start_abs_idx: int, end_abs_idx: int) -> List[Point]:
        if start_abs_idx > end_abs_idx:
            return []
        rel_start = max(0, start_abs_idx - self.cache_offset)
        rel_end = end_abs_idx - self.cache_offset
        return [item[0] for item in self.cache[rel_start:rel_end + 1]]

    def _prune_cache(self, new_start_abs_idx: int):
        if new_start_abs_idx > self.cache_offset:
            idx_to_remove = new_start_abs_idx - self.cache_offset
            self.cache = self.cache[idx_to_remove:]
            self.cache_offset = new_start_abs_idx

    def _create_stop(self, points: List[Point]) -> Stop:
        if not points:
            raise ValueError("Empty points for Stop")
        lat = sum(p.lat for p in points) / len(points)
        lon = sum(p.lon for p in points) / len(points)
        centroid = Point(
            lat=lat, lon=lon,
            timestamp=points[0].timestamp,
            obj_id=points[0].obj_id
        )
        return Stop(points=points, centroid=centroid)

    def process_point(self, p_c: Point) -> List[Segment]:
        """
        Processes a newly arrived point, updating states and emitting finished sub-trajectories.
        """
        segments = []
        
        if not self.cache:
            self.origin_lat = math.radians(p_c.lat)
            self.origin_lon = math.radians(p_c.lon)
            
        # Calculate local grid
        dx_meters = (math.radians(p_c.lon) - self.origin_lon) * 6371000.0 * math.cos(self.origin_lat)
        dy_meters = (math.radians(p_c.lat) - self.origin_lat) * 6371000.0
        gx_c = int(dx_meters // self.g)
        gy_c = int(dy_meters // self.g)
        
        self.cache.append((p_c, gx_c, gy_c))
        c = self.cache_offset + len(self.cache) - 1
        
        # 1. Indexed Stay Point Detection (Alg 1)
        Is = None
        Ie = c
        i = c - 1
        
        while i >= self.cache_offset:
            p_i, gx_i, gy_i = self._get_cached_item(i)
            delta_x = abs(gx_i - gx_c)
            delta_y = abs(gy_i - gy_c)

            if (delta_x + 1) ** 2 + (delta_y + 1) ** 2 <= self.threshold_sq:
                i -= 1 # Confirmed Area
            elif max(0, delta_x - 1) ** 2 + max(0, delta_y - 1) ** 2 > self.threshold_sq:
                i += 1 # Pruned Area -> Out of bound, restore i to first valid
                break
            elif local_distance(p_c, p_i) <= self.max_eps:
                i -= 1 # Exact check satisfying constraint
            else:
                i += 1 # Exact check violating
                break
                
        if i < self.cache_offset:
            i = self.cache_offset
            
        if i <= c:
            p_i_point = self._get_point(i)
            duration_secs = (p_c.timestamp - p_i_point.timestamp).total_seconds()
            if duration_secs >= self.min_duration_seconds:
                Is = i

        # 2. Trajectory Segmentation Handling
        if Is is not None:
            # Case 1: Stay point is formed
            if self.current_sp_start is not None:
                if Is <= self.current_sp_end:
                    # Case 1.2: Intersected, merge them
                    self.current_sp_end = Ie
                else:
                    # Case 1.1: Separated, flush first stay point and in-between move
                    sp1_points = self._get_points(self.current_sp_start, self.current_sp_end)
                    segments.append(self._create_stop(sp1_points))
                    
                    move_points = self._get_points(self.current_sp_end + 1, Is - 1)
                    if move_points:
                        segments.append(Move(points=move_points))
                        
                    self.current_sp_start = Is
                    self.current_sp_end = Ie
                    self._prune_cache(Is)
            else:
                # Case 1.3: Only one stay point. Flush anything before it as a move.
                move_points = self._get_points(self.cache_offset, Is - 1)
                if move_points:
                    segments.append(Move(points=move_points))
                    
                self.current_sp_start = Is
                self.current_sp_end = Ie
                self._prune_cache(Is)
        else:
            # Case 2: New point does NOT form stay point
            if self.current_sp_start is not None:
                p_Ie = self._get_point(self.current_sp_end)
                if local_distance(p_c, p_Ie) > self.max_eps:
                    # Case 2.1: Far away from last stay point. Flush stay point.
                    sp1_points = self._get_points(self.current_sp_start, self.current_sp_end)
                    segments.append(self._create_stop(sp1_points))
                    
                    self._prune_cache(self.current_sp_end + 1)
                    self.current_sp_start = None
                    self.current_sp_end = None
            # Else Case 2.2/2.3: do nothing.

        return segments

    def flush(self) -> List[Segment]:
        """
        Emits remaining cached segments upon termination of stream.
        """
        segments = []
        if self.current_sp_start is not None:
            sp_points = self._get_points(self.current_sp_start, self.current_sp_end)
            segments.append(self._create_stop(sp_points))
            move_points = self._get_points(self.current_sp_end + 1, self.cache_offset + len(self.cache) - 1)
            if move_points:
                segments.append(Move(points=move_points))
        else:
            move_points = self._get_points(self.cache_offset, self.cache_offset + len(self.cache) - 1)
            if move_points:
                segments.append(Move(points=move_points))
                
        self.cache = []
        self.current_sp_start = None
        self.current_sp_end = None
        return segments

    def process(self, trajectory: List[Point]) -> List[Segment]:
        """
        Batch-processing helper for testing/benchmarking.
        """
        segments = []
        for p in trajectory:
            segments.extend(self.process_point(p))
        segments.extend(self.flush())
        return segments
