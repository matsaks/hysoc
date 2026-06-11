"""Immutable GPS fix primitive."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Point:
    """Immutable single GPS fix (lat, lon, timestamp, obj_id, optional road_id)."""
    lat: float
    lon: float
    timestamp: datetime
    obj_id: str
    road_id: str | int | None = None

    @property
    def tuple(self):
        return (self.lat, self.lon, self.timestamp, self.obj_id, self.road_id)

