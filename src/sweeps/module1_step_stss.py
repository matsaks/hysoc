"""Module I sweep: STEP versus STSS over the (eps, T) grid."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.point import Point
from core.segment import Segment
from engines.step import STEPSegmenter
from engines.stss import STSSOracle
from eval.latency import LatencyStats, measure_latency
from eval.segmentation import segment_counts, stop_f1
from sweeps._common import (
    DATASET_REGISTRY,
    DEFAULT_DATASET,
    discover_calibration_csvs,
    load_trajectory,
    make_sweep_output_dir,
    resolve_dataset_dir,
    summarise,
    write_aggregated_csv,
    write_per_trajectory_csv,
    write_run_config,
)


SWEEP_NAME = "module1_step_stss"

EPS_GRID_M: list[float] = [10.0, 15.0, 20.0, 30.0, 50.0]
T_GRID_S: list[float] = [10.0, 15.0, 30.0, 60.0]

CONFIG_COLS = ["eps_m", "t_s", "min_samples"]
METRIC_COLS = [
    "n_points",
    "step_n_stops",
    "step_n_moves",
    "step_mean_stop_s",
    "step_mean_move_pts",
    "step_wall_s",
    "step_wall_p25_s",
    "step_wall_p75_s",
    "stss_n_stops",
    "stss_n_moves",
    "stss_mean_stop_s",
    "stss_mean_move_pts",
    "stss_wall_s",
    "stss_wall_p25_s",
    "stss_wall_p75_s",
    "f1",
    "precision",
    "recall",
    "matched_iou_mean",
]

F1_FLOOR_MEDIAN: float = 0.6
MIN_MEDIAN_STEP_STOPS: float = 2.0
MIN_MEDIAN_STEP_MOVES: float = 2.0

F1_TIE_TOLERANCE: float = 0.05

# Reduced repeat count: STSS dominates sweep wall-clock and extra repeats add no precision.
STSS_LATENCY_REPEATS: int = 2


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
    step_wall_p25_s: float
    step_wall_p75_s: float
    stss_n_stops: int
    stss_n_moves: int
    stss_mean_stop_s: float
    stss_mean_move_pts: float
    stss_wall_s: float
    stss_wall_p25_s: float
    stss_wall_p75_s: float
    f1: float
    precision: float
    recall: float
    matched_iou_mean: float


def derive_min_samples(t_s: float) -> int:
    """Derive STSS min_samples from the dwell window, floored at 5."""
    return max(5, round(t_s * 0.5))


def run_step(traj: list[Point], eps: float, t: float) -> tuple[list[Segment], LatencyStats]:
    """Run STEP once for output, then time it under ``measure_latency``."""
    out = STEPSegmenter(max_eps=eps, min_duration_seconds=t).process(traj)

    def _work() -> None:
        # Fresh segmenter per repeat to avoid caching effects.
        STEPSegmenter(max_eps=eps, min_duration_seconds=t).process(traj)

    return out, measure_latency(_work, n_elements=len(traj))


def run_stss(
    traj: list[Point], eps: float, t: float, min_samples: int
) -> tuple[list[Segment], LatencyStats]:
    """Run STSS once for output, then time it under ``measure_latency``."""
    out = STSSOracle(
        min_samples=min_samples, max_eps=eps, min_duration_seconds=t
    ).process(traj)

    def _work() -> None:
        STSSOracle(
            min_samples=min_samples, max_eps=eps, min_duration_seconds=t
        ).process(traj)

    return out, measure_latency(_work, n_elements=len(traj), repeats=STSS_LATENCY_REPEATS)


def summarise_run(
    traj_name: str,
    eps: float,
    t: float,
    min_samples: int,
    n_points: int,
    step_segs: list[Segment],
    step_latency: LatencyStats,
    stss_segs: list[Segment],
    stss_latency: LatencyStats,
) -> RunResult:
    step_counts = segment_counts(step_segs)
    stss_counts = segment_counts(stss_segs)
    f1 = stop_f1(predicted=step_segs, ground_truth=stss_segs, temporal_iou_threshold=0.5)

    # Per-element microseconds back to total wall-clock seconds for the METRIC_COLS semantics.
    def _wall_s(us_per_point: float) -> float:
        return us_per_point * n_points / 1e6

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
        step_wall_s=_wall_s(step_latency.median_us),
        step_wall_p25_s=_wall_s(step_latency.p25_us),
        step_wall_p75_s=_wall_s(step_latency.p75_us),
        stss_n_stops=int(stss_counts["n_stops"]),
        stss_n_moves=int(stss_counts["n_moves"]),
        stss_mean_stop_s=float(stss_counts["mean_stop_duration_s"]),
        stss_mean_move_pts=float(stss_counts["mean_move_points"]),
        stss_wall_s=_wall_s(stss_latency.median_us),
        stss_wall_p25_s=_wall_s(stss_latency.p25_us),
        stss_wall_p75_s=_wall_s(stss_latency.p75_us),
        f1=f1.f1,
        precision=f1.precision,
        recall=f1.recall,
        matched_iou_mean=f1.matched_iou_mean,
    )


def run_sweep(trajectories: list[tuple[str, list[Point]]]) -> list[RunResult]:
    results: list[RunResult] = []
    total_configs = len(EPS_GRID_M) * len(T_GRID_S)
    total_runs = len(trajectories) * total_configs
    overall_start = time.perf_counter()
    done = 0

    for eps in EPS_GRID_M:
        for t in T_GRID_S:
            min_samples = derive_min_samples(t)
            for traj_name, pts in trajectories:
                if len(pts) < 5:
                    continue
                step_segs, step_latency = run_step(pts, eps, t)
                stss_segs, stss_latency = run_stss(pts, eps, t, min_samples)
                results.append(
                    summarise_run(
                        traj_name,
                        eps,
                        t,
                        min_samples,
                        len(pts),
                        step_segs,
                        step_latency,
                        stss_segs,
                        stss_latency,
                    )
                )
                done += 1
                if done % 25 == 0 or done == total_runs:
                    elapsed = time.perf_counter() - overall_start
                    eta = elapsed / done * (total_runs - done) if done else 0
                    print(
                        f"  [{done:>5}/{total_runs}] eps={eps:>4.0f}m T={t:>3.0f}s "
                        f"elapsed={elapsed:7.1f}s eta={eta:7.1f}s"
                    )

    return results


def aggregate(results: list[RunResult]) -> list[dict]:
    by_cfg: dict[tuple[float, float, int], list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault((r.eps_m, r.t_s, r.min_samples), []).append(r)

    aggregated: list[dict] = []
    for (eps_m, t_s, min_samples), cfg_rows in sorted(by_cfg.items()):
        out: dict = {
            "eps_m": eps_m,
            "t_s": t_s,
            "min_samples": min_samples,
            "n_trajectories": len(cfg_rows),
        }
        for metric in METRIC_COLS:
            values = [getattr(r, metric) for r in cfg_rows]
            out.update(summarise(values, metric))

        out["passes_plan_heuristic"] = int(
            out["step_n_stops_median"] >= MIN_MEDIAN_STEP_STOPS
            and out["step_n_moves_median"] >= MIN_MEDIAN_STEP_MOVES
            and out["f1_median"] >= F1_FLOOR_MEDIAN
        )
        out["step_ms_per_point_median"] = (
            out["step_wall_s_median"] / out["n_points_median"] * 1000.0
            if out["n_points_median"] > 0
            else 0.0
        )
        aggregated.append(out)

    return aggregated


def select_winner(aggregated: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the highest-segment-count config among those within F1_TIE_TOLERANCE of the best F1."""
    feasible = [r for r in aggregated if r["passes_plan_heuristic"] == 1]
    if not feasible:
        return None, []

    best_f1 = max(float(r["f1_median"]) for r in feasible)
    tied = [
        r for r in feasible
        if best_f1 - float(r["f1_median"]) <= F1_TIE_TOLERANCE
    ]
    others = [r for r in feasible if r not in tied]

    def _segment_richness(r: dict) -> float:
        return float(r["step_n_stops_median"]) + float(r["step_n_moves_median"])

    tied_sorted = sorted(
        tied,
        key=lambda r: (
            -_segment_richness(r),
            float(r["eps_m"]),
            float(r["t_s"]),
        ),
    )
    others_sorted = sorted(
        others,
        key=lambda r: (-float(r["f1_median"]), float(r["eps_m"]), float(r["t_s"])),
    )
    return tied_sorted[0], tied_sorted + others_sorted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY),
        default=DEFAULT_DATASET,
        help="Calibration dataset to sweep on (default: nyc).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap number of trajectories (smoke tests).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output base directory (defaults to results/sweeps/).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared timestamp folder name; defaults to a fresh MMDD_HHMM.",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    files = discover_calibration_csvs(args.dataset, args.max_trajectories)
    out_base = Path(args.out_dir) if args.out_dir else None
    out_dir = make_sweep_output_dir(SWEEP_NAME, run_id=args.run_id, base=out_base)

    print(f"[{SWEEP_NAME}] Output: {out_dir}")
    print(
        f"[{SWEEP_NAME}] Loading {len(files)} trajectories from {dataset_dir}; "
        f"running {len(EPS_GRID_M) * len(T_GRID_S)} configs."
    )

    trajectories: list[tuple[str, list[Point]]] = []
    for csv_path in files:
        pts = load_trajectory(csv_path)
        trajectories.append((csv_path.stem, pts))
    print(
        f"[{SWEEP_NAME}] Loaded {len(trajectories)} trajectories, "
        f"{sum(len(p) for _, p in trajectories):,} points total."
    )

    results = run_sweep(trajectories)
    aggregated = aggregate(results)
    winner, _ = select_winner(aggregated)

    aggregated.sort(
        key=lambda r: (
            -int(r["passes_plan_heuristic"]),
            -float(r["f1_median"]),
            float(r["eps_m"]),
            float(r["t_s"]),
        )
    )

    write_per_trajectory_csv(results, out_dir / "per_trajectory.csv")
    write_aggregated_csv(aggregated, out_dir / "aggregated.csv")
    write_run_config(
        {
            "sweep_name": SWEEP_NAME,
            "dataset": str(dataset_dir),
            "dataset_key": args.dataset,
            "n_trajectories": len(trajectories),
            "eps_grid_m": EPS_GRID_M,
            "t_grid_s": T_GRID_S,
            "min_samples_rule": "max(5, round(T * 0.5))",
            "selection": {
                "f1_floor_median": F1_FLOOR_MEDIAN,
                "min_median_step_stops": MIN_MEDIAN_STEP_STOPS,
                "min_median_step_moves": MIN_MEDIAN_STEP_MOVES,
                "f1_tie_tolerance": F1_TIE_TOLERANCE,
                "rule": (
                    "highest median segment count (stops + moves) among "
                    "feasible configurations whose median F1 is within "
                    "f1_tie_tolerance of the best; final tiebreak smaller (eps, T)"
                ),
            },
        },
        out_dir / "run_config.json",
    )

    print(f"[{SWEEP_NAME}] Wrote {len(results)} per-trajectory rows.")
    print(f"[{SWEEP_NAME}] Wrote {len(aggregated)} aggregated config rows.")
    if winner is not None:
        print(
            f"[{SWEEP_NAME}] Winner: eps_m={winner['eps_m']:g}, t_s={winner['t_s']:g} "
            f"(median F1 {winner['f1_median']:.2f})"
        )
    else:
        print(f"[{SWEEP_NAME}] No configuration passed the heuristic.")


if __name__ == "__main__":
    main()
