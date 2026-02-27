from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Point:
    """
    Represents a single GPS fix (x, y, t).
    frozen=True makes the class immutable, which is safer for streaming logic"
    """
    lat: float
    lon: float
    timestamp: datetime
    obj_id: str
    road_id: str | int | None = None

    @property
    def tuple(self):
        return (self.lat, self.lon, self.timestamp, self.obj_id, self.road_id)

