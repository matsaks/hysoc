"""Trajectory segment primitives (Stop and Move)."""

from dataclasses import dataclass, field
from datetime import datetime
from .point import Point


@dataclass(frozen=True)
class Segment:
    """Base class for a trajectory segment (Stop or Move) holding a point sequence."""
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
    """A Stop segment with an optional representative centroid."""
    centroid: Point | None = None

@dataclass(frozen=True)
class Move(Segment):
    """A Move segment."""
    pass
