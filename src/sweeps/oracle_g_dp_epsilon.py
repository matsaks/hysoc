"""Oracle-G DP epsilon sweep on STSS-segmented Move segments."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STSS_MAX_EPS_METERS,
    STSS_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from core.point import Point
from core.segment import Move, Stop
from engines.dp import DouglasPeuckerCompressor
from engines.stss import STSSOracle
from eval.latency import LatencyStats, measure_latency
from eval.sed import calculate_sed_stats
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


SWEEP_NAME = "oracle_g_dp_epsilon"

DP_EPSILON_GRID_M: list[float] = [
    1.0,
    2.5,
    5.0,
    6.0,
    7.0,
    7.5,
    8.0,
    9.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
]

CONFIG_COLS = ["dp_epsilon_m"]
METRIC_COLS = [
    "n_points",
    "n_moves",
    "n_stops",
    "n_move_points",
    "n_move_keypoints",
    "move_cr",
    "move_mean_sed_m",
    "move_max_sed_m",
    "move_latency_us_per_point",
]

SED_BUDGET_M: float = STOP_MAX_EPS_METERS


@dataclass
class RunResult:
    trajectory: str
    dp_epsilon_m: float
    n_points: int
    n_moves: int
    n_stops: int
    n_move_points: int
    n_move_keypoints: int
    move_cr: float
    move_mean_sed_m: float
    move_max_sed_m: float
    move_latency_us_per_point: float
    move_latency_p25_us_per_point: float
    move_latency_p75_us_per_point: float


def segment_with_stss(points: list[Point]) -> list:
    seg = STSSOracle(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STSS_MAX_EPS_METERS,
        min_duration_seconds=STSS_MIN_DURATION_SECONDS,
    )
    return seg.process(points)


def evaluate_move_compression(
    moves: list[Move], dp_epsilon_m: float
) -> tuple[int, int, list[float], LatencyStats]:
    """Compress every Move with DP(epsilon), measure per-point SED, and time compression."""
    dp = DouglasPeuckerCompressor(epsilon_meters=dp_epsilon_m)

    total_input = 0
    total_output = 0
    sed_errors: list[float] = []

    for m in moves:
        keypoints = dp.compress(m.points)
        total_input += len(m.points)
        total_output += len(keypoints)
        if keypoints and m.points:
            stats = calculate_sed_stats(m.points, keypoints)
            sed_errors.extend(stats["sed_errors"])

    def _compress_only() -> None:
        for m in moves:
            dp.compress(m.points)

    latency = measure_latency(_compress_only, n_elements=total_input)
    return total_input, total_output, sed_errors, latency


def summarise_run(
    traj_name: str,
    dp_epsilon_m: float,
    n_points: int,
    n_moves: int,
    n_stops: int,
    total_input: int,
    total_output: int,
    latency: LatencyStats,
    sed_errors: list[float],
) -> RunResult:
    move_cr = total_input / total_output if total_output > 0 else float("nan")
    if sed_errors:
        move_mean_sed = sum(sed_errors) / len(sed_errors)
        move_max_sed = max(sed_errors)
    else:
        move_mean_sed = float("nan")
        move_max_sed = float("nan")

    return RunResult(
        trajectory=traj_name,
        dp_epsilon_m=dp_epsilon_m,
        n_points=n_points,
        n_moves=n_moves,
        n_stops=n_stops,
        n_move_points=total_input,
        n_move_keypoints=total_output,
        move_cr=move_cr,
        move_mean_sed_m=move_mean_sed,
        move_max_sed_m=move_max_sed,
        move_latency_us_per_point=latency.median_us,
        move_latency_p25_us_per_point=latency.p25_us,
        move_latency_p75_us_per_point=latency.p75_us,
    )


def run_sweep(trajectories: list[tuple[str, list[Point]]]) -> list[RunResult]:
    # Segment each trajectory once; STSS is invariant under DP epsilon.
    cached: list[tuple[str, int, int, int, list[Move]]] = []
    seg_start = time.perf_counter()
    for traj_name, pts in trajectories:
        if len(pts) < 5:
            continue
        segments = segment_with_stss(pts)
        moves = [s for s in segments if isinstance(s, Move)]
        n_stops = sum(1 for s in segments if isinstance(s, Stop))
        cached.append((traj_name, len(pts), n_stops, len(moves), moves))

    total_move_points = sum(len(m.points) for _, _, _, _, ms in cached for m in ms)
    print(
        f"[{SWEEP_NAME}] Segmented {len(cached)} trajectories with STSS in "
        f"{time.perf_counter() - seg_start:.1f}s "
        f"({total_move_points:,} move points total).",
        flush=True,
    )

    results: list[RunResult] = []
    overall_start = time.perf_counter()
    total_runs = sum(1 for _, _, _, n_moves, _ in cached if n_moves > 0) * len(
        DP_EPSILON_GRID_M
    )
    done = 0

    for eps in DP_EPSILON_GRID_M:
        for traj_name, n_points, n_stops, n_moves, moves in cached:
            if not moves:
                continue
            total_input, total_output, sed_errors, latency = evaluate_move_compression(
                moves, eps
            )
            results.append(
                summarise_run(
                    traj_name,
                    eps,
                    n_points,
                    n_moves,
                    n_stops,
                    total_input,
                    total_output,
                    latency,
                    sed_errors,
                )
            )
            done += 1
            if done % 100 == 0 or done == total_runs:
                elapsed_total = time.perf_counter() - overall_start
                eta = elapsed_total / done * (total_runs - done) if done else 0
                print(
                    f"  [{done:>5}/{total_runs}] eps={eps:>4.1f}m "
                    f"elapsed={elapsed_total:7.1f}s eta={eta:7.1f}s"
                )

    return results


def aggregate(results: list[RunResult]) -> list[dict]:
    by_cfg: dict[float, list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault(r.dp_epsilon_m, []).append(r)

    aggregated: list[dict] = []
    for eps, cfg_rows in sorted(by_cfg.items()):
        out: dict = {
            "dp_epsilon_m": eps,
            "n_trajectories": len(cfg_rows),
        }
        for metric in METRIC_COLS:
            values = [getattr(r, metric) for r in cfg_rows]
            out.update(summarise(values, metric))

        median_mean_sed = out["move_mean_sed_m_median"]
        out["passes_sed_budget"] = int(
            isinstance(median_mean_sed, float)
            and not math.isnan(median_mean_sed)
            and median_mean_sed <= SED_BUDGET_M
        )
        aggregated.append(out)

    return aggregated


def select_winner(aggregated: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the highest median move CR among configs within SED_BUDGET_M, tiebreak by latency."""
    feasible = [r for r in aggregated if r["passes_sed_budget"] == 1]
    if not feasible:
        return None, []
    feasible_sorted = sorted(
        feasible,
        key=lambda r: (
            -float(r["move_cr_median"]),
            float(r["move_latency_us_per_point_median"]),
        ),
    )
    return feasible_sorted[0], feasible_sorted


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
        f"running {len(DP_EPSILON_GRID_M)} configs."
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
            -int(r["passes_sed_budget"]),
            -float(r["move_cr_median"])
            if not math.isnan(float(r["move_cr_median"]))
            else 0.0,
            float(r["dp_epsilon_m"]),
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
            "dp_epsilon_grid_m": DP_EPSILON_GRID_M,
            "stss_max_eps_m": STSS_MAX_EPS_METERS,
            "stss_min_duration_s": STSS_MIN_DURATION_SECONDS,
            "stss_min_samples": STSS_MIN_SAMPLES,
            "selection": {
                "rule": (
                    "highest median move CR among configurations whose "
                    "median move-segment mean SED <= sed_budget_m "
                    "(error-bounded line simplification, SQUISH-E(mu) / "
                    "Lin et al. 2021)"
                ),
                "tiebreak": "lower median latency",
                "sed_budget_m": SED_BUDGET_M,
            },
        },
        out_dir / "run_config.json",
    )

    print(f"[{SWEEP_NAME}] Wrote {len(results)} per-trajectory rows.")
    print(f"[{SWEEP_NAME}] Wrote {len(aggregated)} aggregated config rows.")
    if winner is not None:
        print(
            f"[{SWEEP_NAME}] Winner: eps={winner['dp_epsilon_m']:g} m "
            f"(median move CR {winner['move_cr_median']:.2f})"
        )
    else:
        print(f"[{SWEEP_NAME}] No configuration passed the SED budget.")


if __name__ == "__main__":
    main()
