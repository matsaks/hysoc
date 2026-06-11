"""Module III HYSOC-N sweep: TRACE (gamma, k) on Move segments."""

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
    STOP_MIN_DURATION_SECONDS,
)
from core.compression import BYTES_PER_POINT
from core.point import Point
from core.segment import Move, Stop
from core.trace_config import TraceConfig
from engines.step import STEPSegmenter
from engines.trace import TraceCompressor
from eval.latency import LatencyStats
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


SWEEP_NAME = "module4_hysoc_n_trace"

GAMMA_GRID_MPS: list[float] = [2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
K_GRID: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Floor on median retention; below it TRACE collapses to STC-equivalent road-transition keypoints.
RETENTION_FLOOR: float = 0.10

CONFIG_COLS = ["gamma_mps", "k"]
METRIC_COLS = [
    "n_points",
    "n_moves",
    "n_stops",
    "n_move_points",
    "n_move_keypoints",
    "move_cr",
    "move_retention_ratio",
    "e_match_rate",
    "v_match_rate",
    "move_latency_us_per_point",
]


@dataclass
class RunResult:
    trajectory: str
    gamma_mps: float
    k: int
    n_points: int
    n_moves: int
    n_stops: int
    n_move_points: int
    n_move_keypoints: int
    move_cr: float
    move_retention_ratio: float
    e_match_rate: float
    v_match_rate: float
    move_latency_us_per_point: float
    move_latency_p25_us_per_point: float
    move_latency_p75_us_per_point: float


def segment_with_step(points: list[Point]) -> list:
    seg = STEPSegmenter(
        max_eps=STOP_MAX_EPS_METERS, min_duration_seconds=STOP_MIN_DURATION_SECONDS
    )
    return seg.process(points)


def road_jaccard_check(moves: list[Move], gamma: float, k: int) -> float:
    """One-time sanity check on TRACE's structural Jaccard claim."""
    cfg = TraceConfig(gamma=gamma, k=k)
    compressor = TraceCompressor(cfg)
    input_roads: set = set()
    output_roads: set = set()
    for m in moves:
        for p in m.points:
            if p.road_id is not None:
                input_roads.add(p.road_id)
        result = compressor.compress(m.points)
        for p in result.keypoints:
            if p.road_id is not None:
                output_roads.add(p.road_id)
    union = input_roads | output_roads
    if not union:
        return float("nan")
    return len(input_roads & output_roads) / len(union)


def evaluate_move_compression(
    moves: list[Move],
    compressor: TraceCompressor,
) -> tuple[int, int, int, int, int, int, int, LatencyStats]:
    """Compress every Move and accumulate per-trajectory byte/match accounting."""
    total_input = 0
    total_keypoints = 0
    total_encoded_bytes = 0
    e_factors_total = 0
    e_matches_total = 0
    v_factors_total = 0
    v_matches_total = 0

    # Shared H mutates across trajectories, so timing is a single wall-clock pass.
    t_start = time.perf_counter_ns()
    for m in moves:
        result = compressor.compress(m.points)
        total_input += len(m.points)
        total_keypoints += len(result.keypoints)
        total_encoded_bytes += result.encoded_bytes
        diag = compressor.diagnostics
        e_lit = int(diag["literal_factors_e"])
        e_mat = int(diag["match_factors_e"])
        v_lit = int(diag["literal_factors_v"])
        v_mat = int(diag["match_factors_v"])
        e_factors_total += e_lit + e_mat
        e_matches_total += e_mat
        v_factors_total += v_lit + v_mat
        v_matches_total += v_mat
    elapsed_ns = time.perf_counter_ns() - t_start
    if total_input > 0:
        us_per_point = elapsed_ns / 1000.0 / total_input
    else:
        us_per_point = float("nan")
    nan = float("nan")
    latency = LatencyStats(
        n_elements=total_input,
        warmup=0,
        n_repeats=1,
        median_us=us_per_point,
        p25_us=nan,
        p75_us=nan,
        mean_us=us_per_point,
        std_us=nan,
        min_us=us_per_point,
        max_us=us_per_point,
    )

    return (
        total_input,
        total_keypoints,
        total_encoded_bytes,
        e_factors_total,
        e_matches_total,
        v_factors_total,
        v_matches_total,
        latency,
    )


def summarise_run(
    traj_name: str,
    gamma: float,
    k: int,
    n_points: int,
    n_moves: int,
    n_stops: int,
    total_input: int,
    total_keypoints: int,
    total_encoded_bytes: int,
    e_factors: int,
    e_matches: int,
    v_factors: int,
    v_matches: int,
    latency: LatencyStats,
) -> RunResult:
    move_cr = (
        (total_input * BYTES_PER_POINT) / total_encoded_bytes
        if total_encoded_bytes > 0
        else float("nan")
    )
    retention = total_keypoints / total_input if total_input > 0 else float("nan")
    e_rate = e_matches / e_factors if e_factors > 0 else float("nan")
    v_rate = v_matches / v_factors if v_factors > 0 else float("nan")

    return RunResult(
        trajectory=traj_name,
        gamma_mps=gamma,
        k=k,
        n_points=n_points,
        n_moves=n_moves,
        n_stops=n_stops,
        n_move_points=total_input,
        n_move_keypoints=total_keypoints,
        move_cr=move_cr,
        move_retention_ratio=retention,
        e_match_rate=e_rate,
        v_match_rate=v_rate,
        move_latency_us_per_point=latency.median_us,
        move_latency_p25_us_per_point=latency.p25_us,
        move_latency_p75_us_per_point=latency.p75_us,
    )


def run_sweep(
    trajectories: list[tuple[str, list[Point]]],
    shared_h: bool = True,
) -> list[RunResult]:
    # Segment each trajectory once; STEP is invariant under (gamma, k).
    cached: list[tuple[str, int, int, int, list[Move]]] = []
    for traj_name, pts in trajectories:
        if len(pts) < 5:
            continue
        segments = segment_with_step(pts)
        moves = [s for s in segments if isinstance(s, Move)]
        n_stops = sum(1 for s in segments if isinstance(s, Stop))
        cached.append((traj_name, len(pts), n_stops, len(moves), moves))

    total_move_points = sum(len(m.points) for _, _, _, _, ms in cached for m in ms)
    mode = "shared H" if shared_h else "per-trajectory H"
    print(
        f"[{SWEEP_NAME}] Segmented {len(cached)} trajectories with STEP "
        f"({total_move_points:,} move points total). Mode: {mode}.",
        flush=True,
    )

    # One-time sanity check on the structural Jaccard claim; all configs give the same answer.
    sanity_traj = next(((n, ms) for n, _, _, _, ms in cached if ms), None)
    if sanity_traj is not None:
        sanity_name, sanity_moves = sanity_traj
        sanity_jaccard = road_jaccard_check(
            sanity_moves, gamma=GAMMA_GRID_MPS[0], k=K_GRID[0]
        )
        print(
            f"[{SWEEP_NAME}] Road-Jaccard sanity (trajectory {sanity_name}, "
            f"gamma={GAMMA_GRID_MPS[0]:g}, k={K_GRID[0]}): {sanity_jaccard:.4f} "
            f"(expected 1.0 by construction).",
            flush=True,
        )

    results: list[RunResult] = []
    overall_start = time.perf_counter()
    total_configs = len(GAMMA_GRID_MPS) * len(K_GRID)
    total_runs = sum(1 for _, _, _, n_moves, _ in cached if n_moves > 0) * total_configs
    done = 0

    for gamma in GAMMA_GRID_MPS:
        for k in K_GRID:
            # One TraceCompressor per config, shared so reference-set H accumulates across the stream.
            cfg = TraceConfig(gamma=gamma, k=k)
            shared_compressor = TraceCompressor(cfg) if shared_h else None
            for traj_name, n_points, n_stops, n_moves, moves in cached:
                if not moves:
                    continue
                compressor = (
                    shared_compressor
                    if shared_compressor is not None
                    else TraceCompressor(cfg)
                )
                (
                    total_input,
                    total_keypoints,
                    total_encoded_bytes,
                    e_factors,
                    e_matches,
                    v_factors,
                    v_matches,
                    latency,
                ) = evaluate_move_compression(moves, compressor)
                results.append(
                    summarise_run(
                        traj_name,
                        gamma,
                        k,
                        n_points,
                        n_moves,
                        n_stops,
                        total_input,
                        total_keypoints,
                        total_encoded_bytes,
                        e_factors,
                        e_matches,
                        v_factors,
                        v_matches,
                        latency,
                    )
                )
                done += 1
                if done % 100 == 0 or done == total_runs:
                    elapsed_total = time.perf_counter() - overall_start
                    eta = (
                        elapsed_total / done * (total_runs - done) if done else 0
                    )
                    print(
                        f"  [{done:>5}/{total_runs}] gamma={gamma:>4.1f}m/s "
                        f"k={k} elapsed={elapsed_total:7.1f}s "
                        f"eta={eta:7.1f}s"
                    )

    return results


def aggregate(results: list[RunResult]) -> list[dict]:
    by_cfg: dict[tuple[float, int], list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault((r.gamma_mps, r.k), []).append(r)

    aggregated: list[dict] = []
    for (gamma, k), cfg_rows in sorted(by_cfg.items()):
        out: dict = {
            "gamma_mps": gamma,
            "k": k,
            "n_trajectories": len(cfg_rows),
        }
        for metric in METRIC_COLS:
            values = [getattr(r, metric) for r in cfg_rows]
            out.update(summarise(values, metric))

        median_retention = out["move_retention_ratio_median"]
        out["passes_retention_floor"] = int(
            isinstance(median_retention, float)
            and not math.isnan(median_retention)
            and median_retention >= RETENTION_FLOOR
        )
        aggregated.append(out)

    return aggregated


def select_winner(aggregated: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the highest median move CR among configs clearing RETENTION_FLOOR, tiebreak by latency."""
    feasible = [r for r in aggregated if r["passes_retention_floor"] == 1]
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
    parser.add_argument(
        "--no-shared-h",
        action="store_true",
        help=(
            "Ablation: instantiate a fresh TraceCompressor per trajectory "
            "(disables cross-trajectory reference-set H sharing)."
        ),
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    files = discover_calibration_csvs(args.dataset, args.max_trajectories)
    out_base = Path(args.out_dir) if args.out_dir else None
    out_dir = make_sweep_output_dir(SWEEP_NAME, run_id=args.run_id, base=out_base)

    print(f"[{SWEEP_NAME}] Output: {out_dir}")
    print(
        f"[{SWEEP_NAME}] Loading {len(files)} trajectories from {dataset_dir}; "
        f"running {len(GAMMA_GRID_MPS) * len(K_GRID)} configs."
    )

    trajectories: list[tuple[str, list[Point]]] = []
    for csv_path in files:
        pts = load_trajectory(csv_path)
        trajectories.append((csv_path.stem, pts))
    print(
        f"[{SWEEP_NAME}] Loaded {len(trajectories)} trajectories, "
        f"{sum(len(p) for _, p in trajectories):,} points total."
    )

    shared_h = not args.no_shared_h
    results = run_sweep(trajectories, shared_h=shared_h)
    aggregated = aggregate(results)
    winner, _ = select_winner(aggregated)

    aggregated.sort(
        key=lambda r: (
            -int(r["passes_retention_floor"]),
            -float(r["move_cr_median"])
            if not math.isnan(float(r["move_cr_median"]))
            else 0.0,
            float(r["move_latency_us_per_point_median"]),
            float(r["gamma_mps"]),
            int(r["k"]),
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
            "gamma_grid_mps": GAMMA_GRID_MPS,
            "k_grid": K_GRID,
            "step_max_eps_m": STOP_MAX_EPS_METERS,
            "step_min_duration_s": STOP_MIN_DURATION_SECONDS,
            "shared_h_across_trajectories": shared_h,
            "selection": {
                "rule": (
                    "highest median move CR among configurations whose "
                    "median move-segment retention ratio >= retention_floor; "
                    "tiebreak by lower median latency"
                ),
                "retention_floor": RETENTION_FLOOR,
                "note": (
                    "TRACE is structurally lossless on the road-segment set, "
                    "so road-segment Jaccard is 1.0 by construction and is "
                    "not a useful selection axis (a one-time sanity check is "
                    "logged at run time). The retention floor prevents the "
                    "high-gamma collapse to STC-equivalent behaviour."
                ),
            },
        },
        out_dir / "run_config.json",
    )

    print(f"[{SWEEP_NAME}] Wrote {len(results)} per-trajectory rows.")
    print(f"[{SWEEP_NAME}] Wrote {len(aggregated)} aggregated config rows.")
    if winner is not None:
        print(
            f"[{SWEEP_NAME}] Winner: gamma={winner['gamma_mps']:g} m/s, "
            f"k={winner['k']} "
            f"(median move CR {winner['move_cr_median']:.2f}, "
            f"median retention {winner['move_retention_ratio_median']:.3f})"
        )
    else:
        print(
            f"[{SWEEP_NAME}] No configuration cleared the retention floor "
            f"of {RETENTION_FLOOR:g}."
        )


if __name__ == "__main__":
    main()
