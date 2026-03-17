import math
from typing import List
from hysoc.core.point import Point


class DouglasPeuckerCompressor:
    """
    Implements the standard offline Ramer-Douglas-Peucker (DP) line simplification algorithm.
    """

    def __init__(self, epsilon_meters: float = 10.0):
        """
        Initialize the Douglas-Peucker standard compressor.
        
        Args:
            epsilon_meters: Maximum allowed perpendicular distance from the simplified line segment 
                            in meters. Points with deviations greater than this threshold are retained.
        """
        self.epsilon_meters = epsilon_meters

    def _perpendicular_distance(self, point: Point, start: Point, end: Point) -> float:
        """
        Calculates the perpendicular geometric distance from 'point' to the line passing 
        through 'start' and 'end' in meters.
        """
        if start.lat == end.lat and start.lon == end.lon:
            # The line is actually a single point
            d_lat = point.lat - start.lat
            d_lon = point.lon - start.lon
            avg_lat = math.radians((point.lat + start.lat) / 2.0)
            d_lat_m = d_lat * 111320.0
            d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
            return math.sqrt(d_lat_m*d_lat_m + d_lon_m*d_lon_m)
            
        # Standard cross-track distance approximation for short distances
        avg_lat = math.radians((start.lat + end.lat + point.lat) / 3.0)
        
        # Convert all to local metric space centered around (start.lon, start.lat) == (0,0)
        x0 = (point.lon - start.lon) * 111320.0 * math.cos(avg_lat)
        y0 = (point.lat - start.lat) * 111320.0
        
        x2 = (end.lon - start.lon) * 111320.0 * math.cos(avg_lat)
        y2 = (end.lat - start.lat) * 111320.0
        
        # Line from start(0,0) to end(x2, y2). Eq: y2*x - x2*y = 0
        num = abs(y2 * x0 - x2 * y0)
        den = math.sqrt(y2**2 + x2**2)
        
        if den == 0:
            return 0.0
            
        return num / den

    def compress(self, points: List[Point]) -> List[Point]:
        """
        Applies the DP algorithm to recursively simplify the sequence of points.
        
        Args:
            points: Ordered sequence of raw points to simplify.
            
        Returns:
            A simplified, ordered list of preserved points.
        """
        if len(points) <= 2:
            return points

        dmax = 0.0
        index = 0
        end = len(points) - 1

        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d

        if dmax > self.epsilon_meters:
            # Recursive call
            results1 = self.compress(points[:index + 1])
            results2 = self.compress(points[index:])
            
            # Combine, excluding the duplicate point at 'index'
            return results1[:-1] + results2
        else:
            return [points[0], points[end]]
