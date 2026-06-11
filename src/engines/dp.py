"""Douglas-Peucker line simplification compressor."""

import math
from typing import List

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.geo_defaults import METERS_PER_DEGREE_LAT
from core.point import Point


class DouglasPeuckerCompressor:
    """Standard offline Ramer-Douglas-Peucker line simplification."""

    def __init__(self, epsilon_meters: float = DP_DEFAULT_EPSILON_METERS):
        self.epsilon_meters = epsilon_meters

    def _perpendicular_distance(self, point: Point, start: Point, end: Point) -> float:
        """Perpendicular distance from a point to the start-end line, in metres."""
        if start.lat == end.lat and start.lon == end.lon:
            # Degenerate line: start and end coincide.
            d_lat = point.lat - start.lat
            d_lon = point.lon - start.lon
            avg_lat = math.radians((point.lat + start.lat) / 2.0)
            d_lat_m = d_lat * METERS_PER_DEGREE_LAT
            d_lon_m = d_lon * METERS_PER_DEGREE_LAT * math.cos(avg_lat)
            return math.sqrt(d_lat_m * d_lat_m + d_lon_m * d_lon_m)

        avg_lat = math.radians((start.lat + end.lat + point.lat) / 3.0)

        # Local metric space centred at start == (0, 0).
        x0 = (point.lon - start.lon) * METERS_PER_DEGREE_LAT * math.cos(avg_lat)
        y0 = (point.lat - start.lat) * METERS_PER_DEGREE_LAT

        x2 = (end.lon - start.lon) * METERS_PER_DEGREE_LAT * math.cos(avg_lat)
        y2 = (end.lat - start.lat) * METERS_PER_DEGREE_LAT

        num = abs(y2 * x0 - x2 * y0)
        den = math.sqrt(y2**2 + x2**2)

        if den == 0:
            return 0.0

        return num / den

    def compress(self, points: List[Point]) -> List[Point]:
        """Recursively simplify the point sequence."""
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
            results1 = self.compress(points[:index + 1])
            results2 = self.compress(points[index:])

            # Drop the duplicate point shared at the split index.
            return results1[:-1] + results2
        else:
            return [points[0], points[end]]
