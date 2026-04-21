"""
Parameter sweep for STEP vs STSS on the NYC WorldTrace subset.

Sweeps a grid of (eps, T) values shared by STEP and STSS, runs both
algorithms on every trajectory in data/raw/NYC_Top_100_Most_Points/, and
writes per-trajectory + aggregated statistics to
data/processed/param_sweep_step_stss.csv.

Derived STSS `min_samples` per T at 1 Hz:
    min_samples = max(5, round(T * 0.5))
(50 percent GPS-drop tolerance during the dwell window.)

Usage:
    uv run python scripts/demo_27_step_stss_param_sweep.py
    uv run python scripts/demo_27_step_stss_param_sweep.py --max-trajectories 20

Notes:
    STSS (sklearn OPTICS with haversine) scales near O(N log N) per
    trajectory via BallTree; the manual DBSCAN-like backend is quadratic
    and is NOT used here. A single full sweep over 100 trajectories and
    20 configurations can take 30-60 minutes on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from core.point import Point
from core.segment import Segment, Stop
from engines.step import STEPSegmenter
from engines.stss_sklearn import STSSOracleSklearn
from eval.segmentation import segment_counts, stop_f1
from core.stream import TrajectoryStream


NYC_DIR = Path(project_root) / "data" / "raw" / "NYC_Top_100_Most_Points"
OUT_CSV = Path(project_root) / "data" / "processed" / "param_sweep_step_stss.csv"

EPS_GRID_M = [10.0, 15.0, 20.0, 30.0, 50.0]
T_GRID_S = [10.0, 15.0, 30.0, 60.0]


@dataclass
class RunResult:
    trajectory: str
    eps_m: float
    t_s: float
    min_samples: int
    n_points: int
    step_n_stops: int
    step_n_moves: int
    step_mean_stop_s: float
    step_mean_move_pts: float
    step_wall_s: float
    stss_n_stops: int
    stss_n_moves: int
    stss_mean_stop_s: float
    stss_mean_move_pts: float
    stss_wall_s: float
    f1: float
    precision: float
    recall: float
    matched_iou_mean: float


def load_trajectory(csv_path: Path) -> list[Point]:
    """Load a WorldTrace-format NYC CSV into a list of Point."""
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


def derive_min_samples(t_s: float) -> int:
    return max(5, round(t_s * 0.5))


def run_step(trajectory: list[Point], eps: float, t: float) -> tuple[list[Segment], float]:
    seg = STEPSegmenter(max_eps=eps, min_duration_seconds=t)
    start = time.perf_counter()
    out = seg.process(trajectory)
    return out, time.perf_counter() - start


def run_stss(
    trajectory: list[Point], eps: float, t: float, min_samples: int
) -> tuple[list[Segment], float]:
    seg = STSSOracleSklearn(
        min_samples=min_samples, max_eps=eps, min_duration_seconds=t
    )
    start = time.perf_counter()
    out = seg.process(trajectory)
    return out, time.perf_counter() - start


def summarise_run(
    traj_name: str,
    eps: float,
    t: float,
    min_samples: int,
    n_points: int,
    step_segs: list[Segment],
    step_wall: float,
    stss_segs: list[Segment],
    stss_wall: float,
) -> RunResult:
    step_counts = segment_counts(step_segs)
    stss_counts = segment_counts(stss_segs)

    f1_res = stop_f1(
        predicted=step_segs, ground_truth=stss_segs, temporal_iou_threshold=0.5
    )

    return RunResult(
        trajectory=traj_name,
        eps_m=eps,
        t_s=t,
        min_samples=min_samples,
        n_points=n_points,
        step_n_stops=int(step_counts["n_stops"]),
        step_n_moves=int(step_counts["n_moves"]),
        step_mean_stop_s=float(step_counts["mean_stop_duration_s"]),
        step_mean_move_pts=float(step_counts["mean_move_points"]),
        step_wall_s=step_wall,
        stss_n_stops=int(stss_counts["n_stops"]),
        stss_n_moves=int(stss_counts["n_moves"]),
        stss_mean_stop_s=float(stss_counts["mean_stop_duration_s"]),
        stss_mean_move_pts=float(stss_counts["mean_move_points"]),
        stss_wall_s=stss_wall,
        f1=f1_res.f1,
        precision=f1_res.precision,
        recall=f1_res.recall,
        matched_iou_mean=f1_res.matched_iou_mean,
    )


def write_results(results: list[RunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RunResult.__dataclass_fields__.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fields})


def print_aggregate(results: list[RunResult]) -> None:
    """Print a per-configuration median summary table."""
    by_cfg: dict[tuple[float, float], list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault((r.eps_m, r.t_s), []).append(r)

    rows = []
    for (eps, t), rs in sorted(by_cfg.items()):
        rows.append(
            dict(
                eps_m=eps,
                t_s=t,
                n_traj=len(rs),
                median_step_stops=statistics.median(r.step_n_stops for r in rs),
                median_step_moves=statistics.median(r.step_n_moves for r in rs),
                median_stss_stops=statistics.median(r.stss_n_stops for r in rs),
                median_stss_moves=statistics.median(r.stss_n_moves for r in rs),
                median_f1=statistics.median(r.f1 for r in rs),
                median_iou=statistics.median(r.matched_iou_mean for r in rs),
                mean_step_ms_per_point=1000.0
                * statistics.mean(r.step_wall_s / max(r.n_points, 1) for r in rs),
                mean_stss_s=statistics.mean(r.stss_wall_s for r in rs),
            )
        )

    print(
        f"\n{'eps':>5} {'T':>4} {'N':>4} | "
        f"{'step_stops':>10} {'step_moves':>10} | "
        f"{'stss_stops':>10} {'stss_moves':>10} | "
        f"{'F1':>5} {'IoU':>5} | "
        f"{'step_ms/pt':>10} {'stss_s':>8}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['eps_m']:>5.0f} {row['t_s']:>4.0f} {row['n_traj']:>4} | "
            f"{row['median_step_stops']:>10.1f} {row['median_step_moves']:>10.1f} | "
            f"{row['median_stss_stops']:>10.1f} {row['median_stss_moves']:>10.1f} | "
            f"{row['median_f1']:>5.2f} {row['median_iou']:>5.2f} | "
            f"{row['mean_step_ms_per_point']:>10.3f} {row['mean_stss_s']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap number of trajectories (for smoke tests).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_CSV),
        help=f"Output CSV path (default: {OUT_CSV}).",
    )
    args = parser.parse_args()

    files = sorted(NYC_DIR.glob("*.csv"))
    if args.max_trajectories is not None:
        files = files[: args.max_trajectories]

    total_configs = len(EPS_GRID_M) * len(T_GRID_S)
    print(
        f"Loading {len(files)} trajectories from {NYC_DIR}; "
        f"running {total_configs} configs ({len(EPS_GRID_M)} eps x {len(T_GRID_S)} T)."
    )
    print(
        f"Grids: eps_m = {EPS_GRID_M}, T_s = {T_GRID_S}. "
        f"STSS min_samples = max(5, round(T * 0.5))."
    )

    results: list[RunResult] = []
    overall_start = time.perf_counter()

    trajectories = []
    for csv_path in files:
        pts = load_trajectory(csv_path)
        trajectories.append((csv_path.stem, pts))
    print(
        f"Loaded {len(trajectories)} trajectories, "
        f"{sum(len(p) for _, p in trajectories):,} points total."
    )

    done = 0
    total_runs = len(trajectories) * total_configs
    for eps in EPS_GRID_M:
        for t in T_GRID_S:
            min_samples = derive_min_samples(t)
            for traj_name, pts in trajectories:
                if len(pts) < 5:
                    continue
                step_segs, step_wall = run_step(pts, eps, t)
                stss_segs, stss_wall = run_stss(pts, eps, t, min_samples)
                results.append(
                    summarise_run(
                        traj_name,
                        eps,
                        t,
                        min_samples,
                        len(pts),
                        step_segs,
                        step_wall,
                        stss_segs,
                        stss_wall,
                    )
                )
                done += 1
                if done % 25 == 0 or done == total_runs:
                    elapsed = time.perf_counter() - overall_start
                    eta = elapsed / done * (total_runs - done) if done else 0
                    print(
                        f"  [{done:>5}/{total_runs}] "
                        f"eps={eps:>4.0f}m T={t:>3.0f}s "
                        f"elapsed={elapsed:7.1f}s eta={eta:7.1f}s"
                    )

    write_results(results, Path(args.out))
    print(f"\nWrote {len(results)} rows to {args.out}")
    print_aggregate(results)


if __name__ == "__main__":
    main()
