"""Module II sweep: stop-compression strategy comparison on isolated stops."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from constants.segmentation_defaults import (
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from constants.stop_compression_defaults import StopCompressionStrategy
from core.point import Point
from core.segment import Stop
from engines.step import STEPSegmenter
from engines.stop_compressor import StopCompressor
from eval.latency import measure_latency
from eval.sed import calculate_sed_error
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


SWEEP_NAME = "module2_stop_compression"

STRATEGIES: list[StopCompressionStrategy] = [
    StopCompressionStrategy.CENTROID,
    StopCompressionStrategy.MEDOID,
    StopCompressionStrategy.SNAP_TO_NEAREST,
    StopCompressionStrategy.FIRST_POINT,
]

CONFIG_COLS = ["strategy"]
METRIC_COLS = [
    "n_stops",
    "n_stop_points",
    "avg_sed_m",
    "max_sed_m",
    "p95_sed_m",
    "latency_us_per_stop",
    "latency_us_per_point",
]

SED_BUDGET_M: float = STOP_MAX_EPS_METERS

SED_TIE_TOLERANCE_M: float = 1.0


@dataclass
class RunResult:
    trajectory: str
    strategy: str
    n_stops: int
    n_stop_points: int
    avg_sed_m: float
    max_sed_m: float
    p95_sed_m: float
    latency_us_per_stop: float
    latency_us_per_point: float
    latency_p25_us_per_point: float
    latency_p75_us_per_point: float


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile without numpy."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def evaluate_strategy(
    trajectory_name: str,
    stops: list[Stop],
    strategy: StopCompressionStrategy,
) -> RunResult:
    compressor = StopCompressor(strategy=strategy)
    n_stops = len(stops)
    n_stop_points = sum(len(s.points) for s in stops)

    if n_stops == 0:
        nan = float("nan")
        return RunResult(
            trajectory=trajectory_name,
            strategy=strategy.value,
            n_stops=0,
            n_stop_points=0,
            avg_sed_m=nan,
            max_sed_m=nan,
            p95_sed_m=nan,
            latency_us_per_stop=nan,
            latency_us_per_point=nan,
            latency_p25_us_per_point=nan,
            latency_p75_us_per_point=nan,
        )

    compressed = [compressor.compress(stop.points) for stop in stops]

    def _compress_only() -> None:
        for stop in stops:
            compressor.compress(stop.points)

    latency = measure_latency(_compress_only, n_elements=n_stop_points)
    points_per_stop = n_stop_points / n_stops if n_stops > 0 else 0.0
    latency_us_per_stop = (
        latency.median_us * points_per_stop if points_per_stop > 0 else float("nan")
    )

    sed_errors: list[float] = []
    for stop, c_stop in zip(stops, compressed):
        p_start = replace(c_stop.centroid, timestamp=c_stop.start_time)
        p_end = replace(c_stop.centroid, timestamp=c_stop.end_time)
        for p_orig in stop.points:
            sed_errors.append(calculate_sed_error(p_orig, p_start, p_end))

    avg_sed = sum(sed_errors) / len(sed_errors) if sed_errors else float("nan")
    max_sed = max(sed_errors) if sed_errors else float("nan")
    p95_sed = _percentile(sed_errors, 95.0) if sed_errors else float("nan")

    return RunResult(
        trajectory=trajectory_name,
        strategy=strategy.value,
        n_stops=n_stops,
        n_stop_points=n_stop_points,
        avg_sed_m=avg_sed,
        max_sed_m=max_sed,
        p95_sed_m=p95_sed,
        latency_us_per_stop=latency_us_per_stop,
        latency_us_per_point=latency.median_us,
        latency_p25_us_per_point=latency.p25_us,
        latency_p75_us_per_point=latency.p75_us,
    )


def run_sweep(trajectories: list[tuple[str, list[Point]]]) -> list[RunResult]:
    results: list[RunResult] = []
    overall_start = time.perf_counter()
    total_runs = len(trajectories) * len(STRATEGIES)
    done = 0

    for traj_name, pts in trajectories:
        if len(pts) < 5:
            continue
        segmenter = STEPSegmenter(
            max_eps=STOP_MAX_EPS_METERS, min_duration_seconds=STOP_MIN_DURATION_SECONDS
        )
        segments = segmenter.process(pts)
        stops = [s for s in segments if isinstance(s, Stop)]

        for strategy in STRATEGIES:
            results.append(evaluate_strategy(traj_name, stops, strategy))
            done += 1
            if done % 50 == 0 or done == total_runs:
                elapsed = time.perf_counter() - overall_start
                eta = elapsed / done * (total_runs - done) if done else 0
                print(
                    f"  [{done:>5}/{total_runs}] {traj_name} {strategy.value:>16}  "
                    f"elapsed={elapsed:7.1f}s eta={eta:7.1f}s"
                )

    return results


def aggregate(results: list[RunResult]) -> list[dict]:
    by_cfg: dict[str, list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault(r.strategy, []).append(r)

    aggregated: list[dict] = []
    for strategy, cfg_rows in sorted(by_cfg.items()):
        out: dict = {
            "strategy": strategy,
            "n_trajectories": len(cfg_rows),
        }
        for metric in METRIC_COLS:
            values = [getattr(r, metric) for r in cfg_rows]
            out.update(summarise(values, metric))

        median_avg_sed = out["avg_sed_m_median"]
        out["passes_sed_budget"] = int(
            isinstance(median_avg_sed, float)
            and median_avg_sed == median_avg_sed  # NaN check
            and median_avg_sed <= SED_BUDGET_M
        )
        aggregated.append(out)

    return aggregated


def select_winner(aggregated: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the cheapest feasible strategy within SED_TIE_TOLERANCE_M of the best avg SED."""
    feasible = [r for r in aggregated if r["passes_sed_budget"] == 1]
    if not feasible:
        return None, []

    best_avg_sed = min(float(r["avg_sed_m_median"]) for r in feasible)
    tied = [
        r for r in feasible
        if float(r["avg_sed_m_median"]) - best_avg_sed <= SED_TIE_TOLERANCE_M
    ]
    others = [r for r in feasible if r not in tied]

    tied_sorted = sorted(tied, key=lambda r: float(r["latency_us_per_stop_median"]))
    others_sorted = sorted(others, key=lambda r: float(r["avg_sed_m_median"]))

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
        f"comparing {len(STRATEGIES)} strategies."
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
            float(r["avg_sed_m_median"]) if r["avg_sed_m_median"] == r["avg_sed_m_median"] else float("inf"),
            r["strategy"],
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
            "strategies": [s.value for s in STRATEGIES],
            "step_max_eps_m": STOP_MAX_EPS_METERS,
            "step_min_duration_s": STOP_MIN_DURATION_SECONDS,
            "selection": {
                "sed_budget_m": SED_BUDGET_M,
                "sed_tie_tolerance_m": SED_TIE_TOLERANCE_M,
                "rule": (
                    "lowest median latency among feasible strategies whose "
                    "median avg SED is within sed_tie_tolerance_m of the best"
                ),
            },
        },
        out_dir / "run_config.json",
    )

    print(f"[{SWEEP_NAME}] Wrote {len(results)} per-trajectory rows.")
    print(f"[{SWEEP_NAME}] Wrote {len(aggregated)} aggregated config rows.")
    if winner is not None:
        print(
            f"[{SWEEP_NAME}] Winner: {winner['strategy']} "
            f"(median latency {winner['latency_us_per_stop_median']:.2f} us/stop, "
            f"median avg SED {winner['avg_sed_m_median']:.2f} m)"
        )
    else:
        print(f"[{SWEEP_NAME}] No strategy passed the SED budget.")


if __name__ == "__main__":
    main()
