from dataclasses import dataclass, field
from datetime import datetime
from .point import Point

@dataclass(frozen=True)
class Segment:
    """
    Base class for a trajectory segment (Stop or Move).
    Contains a sequence of points.
    """
    points: list[Point] = field(default_factory=list)

    @property
    def start_time(self) -> datetime:
        if not self.points:
            raise ValueError("Segment is empty")
        return self.points[0].timestamp

    @property
    def end_time(self) -> datetime:
        if not self.points:
            raise ValueError("Segment is empty")
        return self.points[-1].timestamp

@dataclass(frozen=True)
class Stop(Segment):
    """
    Represents a Stop segment.
    """
    centroid: Point | None = None

@dataclass(frozen=True)
class Move(Segment):
    """
    Represents a Move segment.
    """
    pass
