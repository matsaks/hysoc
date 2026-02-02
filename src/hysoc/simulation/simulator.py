import abc
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Iterator

from hysoc.core.point import Point


class Simulator(abc.ABC):
    """Abstract base class for streaming simulators."""

    @abc.abstractmethod
    def stream(self) -> Generator[Point, None, None]:
        """Yields Point objects one by one."""
        pass


class TrajectorySimulator(Simulator):
    """
    Simulates a device stream by reading a trajectory CSV file and emitting
    points at a fixed interval with updated timestamps.
    """

    def __init__(
        self,
        file_path: str | Path,
        obj_id: str | None = None,
        interval: float = 1.0,
        start_time: datetime | None = None,
    ):
        """
        Args:
            file_path: Path to the CSV file containing the trajectory.
            obj_id: Identifier for the moving object. If None, derived from filename.
            interval: Time in seconds to wait between emitting points.
            start_time: The simulation time for the first point. Defaults to now.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self._obj_id = obj_id if obj_id else self.file_path.stem
        self._interval = interval
        self._start_time = start_time if start_time else datetime.now()

    def stream(self) -> Generator[Point, None, None]:
        """
        Reads the CSV and yields points.
        The timestamp of each point is calculated relative to start_time
        based on the index and interval.
        """
        with open(self.file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Ensure required columns exist
            if not reader.fieldnames or "latitude" not in reader.fieldnames or "longitude" not in reader.fieldnames:
                 raise ValueError(f"CSV must contain 'latitude' and 'longitude' columns. Found: {reader.fieldnames}")

            for i, row in enumerate(reader):
                current_time = self._start_time + timedelta(seconds=i * self._interval)
                
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except ValueError:
                    # Skip rows with invalid coordinates
                    continue

                point = Point(
                    lat=lat,
                    lon=lon,
                    timestamp=current_time,
                    obj_id=self._obj_id,
                )
                
                yield point
                
                # Wait for the next interval
                time.sleep(self._interval)
