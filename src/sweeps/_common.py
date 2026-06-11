"""Shared loader, summariser, and output-folder helpers for parameter-sweep scripts."""

from __future__ import annotations

# ruff: noqa: E402

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from constants.dataset_paths import (
    CALIBRATION_DIR as NYC_CALIBRATION_DIR,
    PROJECT_ROOT,
    SAN_FRAN_CENTRE_CALIBRATION_DIR,
)
from core.point import Point
from core.stream import TrajectoryStream


DATASET_REGISTRY: dict[str, Path] = {
    "nyc": NYC_CALIBRATION_DIR,
    "sf_centre": SAN_FRAN_CENTRE_CALIBRATION_DIR,
}

DEFAULT_DATASET: str = "nyc"


def resolve_dataset_dir(dataset: str = DEFAULT_DATASET) -> Path:
    """Look up the calibration directory for ``dataset``."""
    try:
        return DATASET_REGISTRY[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}; known datasets: {known}"
        ) from exc


def load_trajectory(csv_path: Path) -> list[Point]:
    """Load a WorldTrace-format CSV into a list of Point."""
    stream = TrajectoryStream(
        filepath=csv_path,
        col_mapping={
            "lat": "latitude",
            "lon": "longitude",
            "timestamp": "time",
            "obj_id": "obj_id",
            "road_id": "osm_way_id",
        },
        default_obj_id=csv_path.stem,
    )
    return list(stream.stream())


def discover_calibration_csvs(
    dataset: str = DEFAULT_DATASET,
    max_trajectories: int | None = None,
) -> list[Path]:
    """Return sorted CSV paths in the named calibration subset, optionally capped."""
    files = sorted(resolve_dataset_dir(dataset).glob("*.csv"))
    if max_trajectories is not None:
        files = files[:max_trajectories]
    return files


SWEEPS_ROOT: Path = PROJECT_ROOT / "results" / "sweeps"
RUN_ID_FORMAT: str = "%m%d_%H%M"


def make_sweep_output_dir(
    sweep_name: str,
    run_id: str | None = None,
    base: Path | None = None,
) -> Path:
    """Create and return ``<base>/<sweep_name>/<run_id>/``."""
    root = base if base is not None else SWEEPS_ROOT
    timestamp = run_id if run_id is not None else datetime.now().strftime(RUN_ID_FORMAT)
    out_dir = root / sweep_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def summarise(values: Iterable[float], prefix: str) -> dict[str, float]:
    """Compute mean / median / std / min / max with NaN cleaning."""
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not clean:
        nan = float("nan")
        return {
            f"{prefix}_mean": nan,
            f"{prefix}_median": nan,
            f"{prefix}_std": nan,
            f"{prefix}_min": nan,
            f"{prefix}_max": nan,
        }
    clean = [float(v) for v in clean]
    return {
        f"{prefix}_mean": statistics.mean(clean),
        f"{prefix}_median": statistics.median(clean),
        f"{prefix}_std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        f"{prefix}_min": min(clean),
        f"{prefix}_max": max(clean),
    }


def write_per_trajectory_csv(rows: Sequence[Any], out_path: Path) -> None:
    """Write a list of dataclass instances (one per (trajectory, config)) to CSV."""
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    if not is_dataclass(rows[0]):
        raise TypeError("rows must be dataclass instances")
    field_names = [f.name for f in fields(rows[0])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_aggregated_csv(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Write a list of aggregated dicts (one per config) to CSV."""
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(config: dict[str, Any], out_path: Path) -> None:
    """Persist the sweep configuration alongside its outputs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
