"""HYSOC-G end-to-end evaluation against Plain DP and Baseline-G on the evaluation sets."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from constants.dataset_paths import (
    EVALUATION_DIR,
    PROJECT_ROOT,
    SAN_FRAN_CENTRE_EVALUATION_DIR,
)
from constants.operating_points import (
    DEFAULT_OPERATING_POINT,
    OPERATING_POINTS,
    OperatingPoint,
)
from core.compression import (
    BYTES_PER_POINT,
    CompressionStrategy,
    HYSOCGConfig,
    SegmentResult,
    TrajectoryResult,
)
from core.point import Point
from core.squish_dp_config import HybridSquishDPConfig
from core.stream import TrajectoryStream
from engines.dp import DouglasPeuckerCompressor
from eval.latency import LatencyStats, measure_latency
from eval.sed import calculate_sed_from_result
from eval.segmentation import stop_f1_from_result
from hysoc import HYSOCGCompressor
from oracle import OracleGCompressor, OracleGConfig
from sweeps._common import (
    RUN_ID_FORMAT,
    make_sweep_output_dir,
    summarise,
    write_aggregated_csv,
    write_per_trajectory_csv,
    write_run_config,
)


EXPERIMENT_NAME = "hysoc_g_eval"

# Baseline-G uses a reduced repeat count: STSS dominates wall-clock and extra repeats add no precision.
LATENCY_REPEATS_PLAIN_DP: int = 5
LATENCY_REPEATS_BASELINE_G: int = 2
LATENCY_REPEATS_HYSOC_G: int = 5
LATENCY_WARMUP: int = 1

PIPELINES: tuple[str, str, str] = ("plain_dp", "baseline_g", "hysoc_g")
PIPELINE_LABELS: dict[str, str] = {
    "plain_dp": "Plain DP",
    "baseline_g": "Baseline-G",
    "hysoc_g": "HYSOC-G",
}


@dataclass
class EvalRow:
    """One row per (dataset, pipeline, trajectory)."""
    dataset: str
    pipeline: str
    obj_id: str
    n_raw_points: int
    n_keypoints: int
    compression_ratio: float
    mean_sed_m: float
    p95_sed_m: float
    max_sed_m: float
    stop_f1: float
    stop_matched_iou_mean: float
    n_stops: int
    n_moves: int
    # Baseline-G reference stop count, copied to all three rows for conditional Stop F1.
    gt_n_stops: int
    latency_median_us_per_point: float
    latency_p25_us: float
    latency_p75_us: float
    latency_mean_us: float
    latency_n_repeats: int


def _load_trajectory(csv_path: Path) -> list[Point]:
    """Load one WorldTrace CSV into a list of Point."""
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


def _discover_csvs(directory: Path, max_files: int) -> list[Path]:
    """Return up to ``max_files`` CSVs from ``directory``, sorted numerically."""
    files = list(directory.glob("*.csv"))

    def _key(p: Path) -> tuple[int, str]:
        stem = p.stem
        return (int(stem), stem) if stem.isdigit() else (10**12, stem)

    files.sort(key=_key)
    if max_files > 0:
        files = files[:max_files]
    return files


def _plain_dp_result(
    points: list[Point], dp: DouglasPeuckerCompressor, obj_id: str
) -> TrajectoryResult:
    """Wrap raw-trajectory DP output as a single-segment ``TrajectoryResult``."""
    keypoints = dp.compress(points)
    seg = SegmentResult(
        kind="move",
        start_time=points[0].timestamp,
        end_time=points[-1].timestamp,
        keypoints=keypoints,
        encoded_bytes=len(keypoints) * BYTES_PER_POINT,
    )
    return TrajectoryResult(
        object_id=obj_id,
        original_points=points,
        segments=[seg],
        strategy=CompressionStrategy.GEOMETRIC,
    )


def _sed_stats(result: TrajectoryResult) -> tuple[float, float, float]:
    """Return (mean, p95, max) SED in metres for the full trajectory."""
    stats = calculate_sed_from_result(result)
    errors = stats.get("sed_errors", [])
    if not errors:
        return 0.0, 0.0, 0.0
    arr = np.asarray(errors, dtype=float)
    return float(stats["average_sed"]), float(np.percentile(arr, 95)), float(stats["max_sed"])


def _row_from(
    dataset: str,
    pipeline: str,
    obj_id: str,
    raw_points: list[Point],
    result: TrajectoryResult,
    latency: LatencyStats,
    reference: TrajectoryResult | None,
    gt_n_stops: int,
) -> EvalRow:
    mean_sed, p95_sed, max_sed = _sed_stats(result)
    if reference is None:
        f1 = float("nan")
        matched_iou = float("nan")
    else:
        fr = stop_f1_from_result(result, reference)
        f1 = float(fr.f1)
        matched_iou = float(fr.matched_iou_mean)
    return EvalRow(
        dataset=dataset,
        pipeline=pipeline,
        obj_id=obj_id,
        n_raw_points=len(raw_points),
        n_keypoints=len(result.keypoints),
        compression_ratio=float(result.compression_ratio),
        mean_sed_m=mean_sed,
        p95_sed_m=p95_sed,
        max_sed_m=max_sed,
        stop_f1=f1,
        stop_matched_iou_mean=matched_iou,
        n_stops=len(result.stops()),
        n_moves=len(result.moves()),
        gt_n_stops=gt_n_stops,
        latency_median_us_per_point=float(latency.median_us),
        latency_p25_us=float(latency.p25_us),
        latency_p75_us=float(latency.p75_us),
        latency_mean_us=float(latency.mean_us),
        latency_n_repeats=int(latency.n_repeats),
    )


def _evaluate_trajectory(
    dataset: str,
    obj_id: str,
    raw_points: list[Point],
    *,
    dp_compressor: DouglasPeuckerCompressor,
    baseline_g_compressor: OracleGCompressor,
    hysoc_config: HYSOCGConfig,
) -> list[EvalRow]:
    """Run all three pipelines on one trajectory and return three rows."""
    n_raw = len(raw_points)

    plain_result = _plain_dp_result(raw_points, dp_compressor, obj_id)
    baseline_g_result = baseline_g_compressor.compress(raw_points, object_id=obj_id)
    hysoc_g_result = HYSOCGCompressor(config=hysoc_config).compress(
        raw_points, object_id=obj_id
    )

    def _work_plain_dp() -> None:
        _plain_dp_result(raw_points, dp_compressor, obj_id)

    def _work_baseline_g() -> None:
        baseline_g_compressor.compress(raw_points, object_id=obj_id)

    def _work_hysoc_g() -> None:
        # STEPSegmenter is stateful, so re-instantiate the orchestrator per repeat.
        HYSOCGCompressor(config=hysoc_config).compress(raw_points, object_id=obj_id)

    lat_plain = measure_latency(
        _work_plain_dp,
        n_elements=n_raw,
        warmup=LATENCY_WARMUP,
        repeats=LATENCY_REPEATS_PLAIN_DP,
    )
    lat_baseline_g = measure_latency(
        _work_baseline_g,
        n_elements=n_raw,
        warmup=LATENCY_WARMUP,
        repeats=LATENCY_REPEATS_BASELINE_G,
    )
    lat_hysoc_g = measure_latency(
        _work_hysoc_g,
        n_elements=n_raw,
        warmup=LATENCY_WARMUP,
        repeats=LATENCY_REPEATS_HYSOC_G,
    )

    gt_n_stops = len(baseline_g_result.stops())
    rows = [
        _row_from(
            dataset, "plain_dp", obj_id, raw_points, plain_result, lat_plain,
            reference=None, gt_n_stops=gt_n_stops,
        ),
        _row_from(
            dataset,
            "baseline_g",
            obj_id,
            raw_points,
            baseline_g_result,
            lat_baseline_g,
            reference=baseline_g_result,  # self-reference → F1 = 1.0 by construction
            gt_n_stops=gt_n_stops,
        ),
        _row_from(
            dataset,
            "hysoc_g",
            obj_id,
            raw_points,
            hysoc_g_result,
            lat_hysoc_g,
            reference=baseline_g_result,
            gt_n_stops=gt_n_stops,
        ),
    ]
    return rows


def _run_dataset(
    dataset_name: str,
    short_name: str,
    input_dir: Path,
    *,
    dp_compressor: DouglasPeuckerCompressor,
    baseline_g_compressor: OracleGCompressor,
    hysoc_config: HYSOCGConfig,
    max_files: int,
) -> list[EvalRow]:
    csv_paths = _discover_csvs(input_dir, max_files)
    print(f"=== {dataset_name} ({short_name}) ===")
    print(f"  input  : {input_dir}")
    print(f"  files  : {len(csv_paths)}")
    rows: list[EvalRow] = []
    for idx, path in enumerate(csv_paths, 1):
        obj_id = path.stem
        points = _load_trajectory(path)
        if len(points) < 2:
            print(f"  [{idx}/{len(csv_paths)}] {obj_id}: skipped (< 2 points)")
            continue
        new_rows = _evaluate_trajectory(
            dataset_name,
            obj_id,
            points,
            dp_compressor=dp_compressor,
            baseline_g_compressor=baseline_g_compressor,
            hysoc_config=hysoc_config,
        )
        rows.extend(new_rows)
        cr_by_pipe = {r.pipeline: r.compression_ratio for r in new_rows}
        sed_by_pipe = {r.pipeline: r.mean_sed_m for r in new_rows}
        print(
            f"  [{idx}/{len(csv_paths)}] {obj_id} ({len(points)} pts): "
            f"plain={cr_by_pipe['plain_dp']:.2f}x/{sed_by_pipe['plain_dp']:.1f}m  "
            f"baseG={cr_by_pipe['baseline_g']:.2f}x/{sed_by_pipe['baseline_g']:.1f}m  "
            f"hyG={cr_by_pipe['hysoc_g']:.2f}x/{sed_by_pipe['hysoc_g']:.1f}m"
        )
    print()
    return rows


METRIC_COLS: tuple[str, ...] = (
    "compression_ratio",
    "mean_sed_m",
    "p95_sed_m",
    "max_sed_m",
    "stop_f1",
    "stop_matched_iou_mean",
    "n_stops",
    "n_moves",
    "n_keypoints",
    "latency_median_us_per_point",
)


def _aggregate(rows: Iterable[EvalRow]) -> list[dict[str, float | str]]:
    """One aggregated row per (dataset, pipeline)."""
    rows = list(rows)
    out: list[dict[str, float | str]] = []
    by_key: dict[tuple[str, str], list[EvalRow]] = {}
    for r in rows:
        by_key.setdefault((r.dataset, r.pipeline), []).append(r)
    for (dataset, pipeline), group in by_key.items():
        agg: dict[str, float | str] = {
            "dataset": dataset,
            "pipeline": pipeline,
            "n_trajectories": float(len(group)),
        }
        for col in METRIC_COLS:
            values = [float(getattr(r, col)) for r in group]
            agg.update(summarise(values, col))
        out.append(agg)
    return out


def _save_boxplot(rows: list[EvalRow], dataset: str, out_path: Path) -> None:
    by_pipe = {p: [] for p in PIPELINES}
    sed_by_pipe = {p: [] for p in PIPELINES}
    lat_by_pipe = {p: [] for p in PIPELINES}
    for r in rows:
        by_pipe[r.pipeline].append(r.compression_ratio)
        sed_by_pipe[r.pipeline].append(r.mean_sed_m)
        lat_by_pipe[r.pipeline].append(r.latency_median_us_per_point)

    labels = [PIPELINE_LABELS[p] for p in PIPELINES]
    colors = ["#bdbdbd", "#64b5f6", "#1565c0"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (axes[0], [by_pipe[p] for p in PIPELINES], "Compression Ratio (byte-based)", "original / encoded", False),
        (axes[1], [sed_by_pipe[p] for p in PIPELINES], "Mean SED (m)", "metres", False),
        (axes[2], [lat_by_pipe[p] for p in PIPELINES], "Latency (us/point, median)", "us/point (log)", True),
    ]
    for ax, data, title, ylabel, log_scale in panels:
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle(
        f"HYSOC-G evaluation - {dataset} ({len(rows)//len(PIPELINES)} trajectories)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_DATASET_SPECS: tuple[tuple[str, str, Path], ...] = (
    ("NYC_Evaluation_1000", "nyc", EVALUATION_DIR),
    ("SanFranCentre_Evaluation_1000", "sf_centre", SAN_FRAN_CENTRE_EVALUATION_DIR),
)


def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        description="HYSOC-G end-to-end evaluation (NYC + SanFranCentre)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[short for _, short, _ in _DATASET_SPECS],
        default=None,
        help="Subset of datasets by short name (default: all).",
    )
    parser.add_argument(
        "--operating-point",
        choices=sorted(OPERATING_POINTS),
        default=DEFAULT_OPERATING_POINT,
        help="Per-city operating point bundling all calibrated parameters (default: nyc).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Cap trajectories per dataset (0 = all). Use a small value for smoke tests.",
    )
    parser.add_argument(
        "--dp-epsilon-meters",
        type=float,
        default=None,
        help="Override the operating point's DP epsilon (rare; default: use op value).",
    )
    parser.add_argument(
        "--squish-capacity",
        type=int,
        default=None,
        help="Override the operating point's SQUISH capacity (rare; default: use op value).",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    op: OperatingPoint = OPERATING_POINTS[args.operating_point]
    eff_dp_eps = args.dp_epsilon_meters if args.dp_epsilon_meters is not None else op.dp_epsilon_m
    eff_squish_capacity = args.squish_capacity if args.squish_capacity is not None else op.squish_capacity

    # Tag the run directory with the operating-point name to keep NYC and SF folders distinct.
    run_id = f"{datetime.now().strftime(RUN_ID_FORMAT)}_op-{op.name}"
    out_dir = make_sweep_output_dir(
        EXPERIMENT_NAME,
        run_id=run_id,
        base=PROJECT_ROOT / "results" / "experiments",
    )
    print(f"output: {out_dir}")
    print(
        f"operating_point: {op.name} (calibrated on {op.calibrated_on})\n"
        f"config: D={op.stop_max_eps_m} m, T={op.stop_min_duration_s} s, "
        f"eps={eff_dp_eps} m, beta={eff_squish_capacity}, "
        f"max_files={args.max_files}\n"
    )

    # Plain DP and Baseline-G are stateless, so reuse one instance each; HYSOC-G is re-instantiated per trajectory.
    dp_compressor = DouglasPeuckerCompressor(epsilon_meters=eff_dp_eps)
    baseline_g_compressor = OracleGCompressor(
        OracleGConfig(
            stss_max_eps_meters=op.stop_max_eps_m,
            stss_min_duration_seconds=op.stop_min_duration_s,
            stss_min_samples=op.stss_min_samples,
            dp_epsilon_meters=eff_dp_eps,
        )
    )
    hysoc_config = HYSOCGConfig(
        stop_max_eps_meters=op.stop_max_eps_m,
        stop_min_duration_seconds=op.stop_min_duration_s,
        move_config=HybridSquishDPConfig(
            capacity=eff_squish_capacity,
            dp_epsilon_meters=eff_dp_eps,
        ),
    )

    requested = set(args.datasets) if args.datasets else None
    rows_by_dataset: dict[str, list[EvalRow]] = {}
    for dataset_name, short, path in _DATASET_SPECS:
        if requested is not None and short not in requested:
            continue
        if not path.is_dir():
            print(f"[warn] dataset directory missing, skipping: {path}")
            continue
        rows = _run_dataset(
            dataset_name,
            short,
            path,
            dp_compressor=dp_compressor,
            baseline_g_compressor=baseline_g_compressor,
            hysoc_config=hysoc_config,
            max_files=args.max_files,
        )
        rows_by_dataset[short] = rows

    for short, rows in rows_by_dataset.items():
        per_csv = out_dir / f"{short}_per_trajectory.csv"
        write_per_trajectory_csv(rows, per_csv)
        print(f"saved per-trajectory: {per_csv}")

    all_rows: list[EvalRow] = [r for rows in rows_by_dataset.values() for r in rows]
    agg = _aggregate(all_rows)
    agg_csv = out_dir / "aggregated.csv"
    write_aggregated_csv(agg, agg_csv)
    print(f"saved aggregated    : {agg_csv}")

    for short, rows in rows_by_dataset.items():
        if not rows:
            continue
        png_path = out_dir / f"comparison_{short}.png"
        _save_boxplot(rows, short, png_path)
        print(f"saved boxplot       : {png_path}")

    run_config: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "operating_point": op.name,
        "operating_point_calibrated_on": op.calibrated_on,
        "operating_point_full": asdict(op),
        "stop_max_eps_meters": op.stop_max_eps_m,
        "stop_min_duration_seconds": op.stop_min_duration_s,
        "dp_epsilon_meters": eff_dp_eps,
        "squish_capacity": eff_squish_capacity,
        "max_files": args.max_files,
        "latency_warmup": LATENCY_WARMUP,
        "latency_repeats": {
            "plain_dp": LATENCY_REPEATS_PLAIN_DP,
            "baseline_g": LATENCY_REPEATS_BASELINE_G,
            "hysoc_g": LATENCY_REPEATS_HYSOC_G,
        },
        "datasets": [
            {"name": name, "short": short, "path": str(path)}
            for name, short, path in _DATASET_SPECS
            if requested is None or short in requested
        ],
    }
    write_run_config(run_config, out_dir / "run_config.json")

    print(f"\nDone. Results in {out_dir}")
    return out_dir


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
