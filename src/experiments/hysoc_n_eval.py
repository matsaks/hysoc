"""HYSOC-N end-to-end evaluation against Plain TRACE and Baseline-N on the evaluation sets."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    HYSOCNConfig,
    SegmentResult,
    TrajectoryResult,
)
from core.point import Point
from core.trace_config import TraceConfig
from dataclasses import asdict
from engines.trace import TraceCompressor
from eval.latency import LatencyStats, measure_latency
from eval.segmentation import (
    road_segment_jaccard_vs_original,
    stop_f1_from_result,
)
from hysoc import HYSOCNCompressor
from oracle import OracleNCompressor, OracleNConfig
from sweeps._common import (
    RUN_ID_FORMAT,
    make_sweep_output_dir,
    summarise,
    write_aggregated_csv,
    write_per_trajectory_csv,
    write_run_config,
)


EXPERIMENT_NAME = "hysoc_n_eval"

# Plain TRACE and HYSOC-N are single-pass because shared H cannot be reset; Baseline-N uses two repeats.
LATENCY_REPEATS_PLAIN_TRACE: int = 1
LATENCY_REPEATS_BASELINE_N: int = 2
LATENCY_REPEATS_HYSOC_N: int = 1
LATENCY_WARMUP: int = 1

PIPELINES: tuple[str, str, str] = ("plain_trace", "baseline_n", "hysoc_n")
PIPELINE_LABELS: dict[str, str] = {
    "plain_trace": "Plain TRACE",
    "baseline_n": "Baseline-N",
    "hysoc_n": "HYSOC-N",
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
    road_jaccard_vs_truth: float
    stop_f1: float
    stop_matched_iou_mean: float
    n_stops: int
    n_moves: int
    # Baseline-N reference stop count, copied to all three rows for conditional Stop F1.
    gt_n_stops: int
    latency_median_us_per_point: float
    latency_p25_us: float
    latency_p75_us: float
    latency_mean_us: float
    latency_n_repeats: int


def _normalise_osm_id(raw: str | None) -> str | None:
    """Normalise WorldTrace ``osm_way_id`` to a plain string id."""
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].split(",")
        if not inner:
            return None
        s = inner[0].strip().strip("'").strip('"')
    return s if s and s != "nan" else None


def _load_trajectory_with_truth(csv_path: Path) -> list[Point]:
    """Load a WorldTrace CSV into a list of ``Point`` with ground-truth road_id."""
    obj_id = csv_path.stem
    points: list[Point] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue
            gt_id = _normalise_osm_id(row.get("osm_way_id"))
            points.append(
                Point(obj_id=obj_id, lat=lat, lon=lon, timestamp=ts, road_id=gt_id)
            )
    return points


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


def _plain_trace_result(
    raw_points: list[Point],
    trace: TraceCompressor,
    obj_id: str,
) -> TrajectoryResult:
    """Compress the raw trajectory via TRACE; wrap as a single-segment result."""
    # original_points stays the full trajectory so CR is computed against the true input size.
    clean = [p for p in raw_points if p.road_id is not None]
    if not clean:
        seg = SegmentResult(
            kind="move",
            start_time=raw_points[0].timestamp,
            end_time=raw_points[-1].timestamp,
            keypoints=[],
            encoded_bytes=0,
        )
    else:
        seg = trace.compress(clean)
    return TrajectoryResult(
        object_id=obj_id,
        original_points=raw_points,
        segments=[seg],
        strategy=CompressionStrategy.NETWORK_SEMANTIC,
    )


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
    if reference is None:
        f1 = float("nan")
        matched_iou = float("nan")
    else:
        fr = stop_f1_from_result(result, reference)
        f1 = float(fr.f1)
        matched_iou = float(fr.matched_iou_mean)
    jaccard = float(road_segment_jaccard_vs_original(result, raw_points))
    return EvalRow(
        dataset=dataset,
        pipeline=pipeline,
        obj_id=obj_id,
        n_raw_points=len(raw_points),
        n_keypoints=len(result.keypoints),
        compression_ratio=float(result.compression_ratio),
        road_jaccard_vs_truth=jaccard,
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


def _single_pass_latency(elapsed_ns: int, n_elements: int) -> LatencyStats:
    """Wrap a single wall-clock measurement as a LatencyStats (IQR fields NaN to mark single-pass)."""
    if n_elements <= 0:
        nan = float("nan")
        return LatencyStats(
            n_elements=n_elements, warmup=0, n_repeats=1,
            median_us=nan, p25_us=nan, p75_us=nan,
            mean_us=nan, std_us=nan, min_us=nan, max_us=nan,
        )
    us_per_point = elapsed_ns / 1000.0 / n_elements
    nan = float("nan")
    return LatencyStats(
        n_elements=n_elements, warmup=0, n_repeats=1,
        median_us=us_per_point, p25_us=nan, p75_us=nan,
        mean_us=us_per_point, std_us=nan,
        min_us=us_per_point, max_us=us_per_point,
    )


def _evaluate_trajectory(
    dataset: str,
    obj_id: str,
    raw_points: list[Point],
    *,
    plain_trace_compressor: TraceCompressor,
    baseline_n_compressor: OracleNCompressor,
    hysoc_n_config: HYSOCNConfig,
    hysoc_n_trace_compressor: TraceCompressor,
) -> list[EvalRow]:
    """Run all three pipelines on one trajectory and return three rows."""
    n_raw = len(raw_points)

    # Plain TRACE: single-pass timing because the shared H precludes repeats.
    t_pt_start = time.perf_counter_ns()
    plain_result = _plain_trace_result(raw_points, plain_trace_compressor, obj_id)
    elapsed_pt = time.perf_counter_ns() - t_pt_start
    lat_plain = _single_pass_latency(elapsed_pt, n_raw)

    # Baseline-N: timed repeats because STSS dominates wall-clock.
    baseline_result = baseline_n_compressor.compress(raw_points, object_id=obj_id)

    def _work_baseline_n() -> None:
        baseline_n_compressor.compress(raw_points, object_id=obj_id)

    lat_baseline = measure_latency(
        _work_baseline_n,
        n_elements=n_raw,
        warmup=LATENCY_WARMUP,
        repeats=LATENCY_REPEATS_BASELINE_N,
    )

    # HYSOC-N: single-pass STEP + STOP + TRACE timing.
    hysoc_compressor = HYSOCNCompressor(
        config=hysoc_n_config,
        trace_compressor=hysoc_n_trace_compressor,
    )
    t_hn_start = time.perf_counter_ns()
    hysoc_result = hysoc_compressor.compress(raw_points, object_id=obj_id)
    elapsed_hn = time.perf_counter_ns() - t_hn_start
    lat_hysoc = _single_pass_latency(elapsed_hn, n_raw)

    gt_n_stops = len(baseline_result.stops())
    return [
        _row_from(
            dataset, "plain_trace", obj_id, raw_points, plain_result, lat_plain,
            reference=None, gt_n_stops=gt_n_stops,
        ),
        _row_from(
            dataset, "baseline_n", obj_id, raw_points, baseline_result,
            lat_baseline,
            reference=baseline_result, gt_n_stops=gt_n_stops,
        ),
        _row_from(
            dataset, "hysoc_n", obj_id, raw_points, hysoc_result, lat_hysoc,
            reference=baseline_result, gt_n_stops=gt_n_stops,
        ),
    ]


def _run_dataset(
    dataset_name: str,
    short_name: str,
    input_dir: Path,
    *,
    op: OperatingPoint,
    trace_config: TraceConfig,
    max_files: int,
) -> list[EvalRow]:
    csv_paths = _discover_csvs(input_dir, max_files)
    print(f"=== {dataset_name} ({short_name}) ===")
    print(f"  input  : {input_dir}")
    print(f"  files  : {len(csv_paths)}")

    # Each pipeline keeps its own shared H so the comparison is fair.
    plain_trace_compressor = TraceCompressor(trace_config)
    hysoc_n_trace_compressor = TraceCompressor(trace_config)

    # Baseline-N is stateless across trajectories, so one shared instance.
    baseline_n_compressor = OracleNCompressor(
        OracleNConfig(
            stss_max_eps_meters=op.stop_max_eps_m,
            stss_min_duration_seconds=op.stop_min_duration_s,
            stss_min_samples=op.stss_min_samples,
        )
    )

    hysoc_n_config = HYSOCNConfig(
        stop_max_eps_meters=op.stop_max_eps_m,
        stop_min_duration_seconds=op.stop_min_duration_s,
        trace_config=trace_config,
    )

    rows: list[EvalRow] = []
    for idx, path in enumerate(csv_paths, 1):
        obj_id = path.stem
        points = _load_trajectory_with_truth(path)
        if len(points) < 2:
            print(f"  [{idx}/{len(csv_paths)}] {obj_id}: skipped (< 2 points)")
            continue
        new_rows = _evaluate_trajectory(
            dataset_name, obj_id, points,
            plain_trace_compressor=plain_trace_compressor,
            baseline_n_compressor=baseline_n_compressor,
            hysoc_n_config=hysoc_n_config,
            hysoc_n_trace_compressor=hysoc_n_trace_compressor,
        )
        rows.extend(new_rows)
        cr_by = {r.pipeline: r.compression_ratio for r in new_rows}
        jac_by = {r.pipeline: r.road_jaccard_vs_truth for r in new_rows}
        print(
            f"  [{idx}/{len(csv_paths)}] {obj_id} ({len(points)} pts): "
            f"pT={cr_by['plain_trace']:.2f}x/J={jac_by['plain_trace']:.2f}  "
            f"bN={cr_by['baseline_n']:.2f}x/J={jac_by['baseline_n']:.2f}  "
            f"hN={cr_by['hysoc_n']:.2f}x/J={jac_by['hysoc_n']:.2f}",
            flush=True,
        )
    print()
    return rows


METRIC_COLS: tuple[str, ...] = (
    "compression_ratio",
    "road_jaccard_vs_truth",
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
    jac_by_pipe = {p: [] for p in PIPELINES}
    lat_by_pipe = {p: [] for p in PIPELINES}
    for r in rows:
        by_pipe[r.pipeline].append(r.compression_ratio)
        if not math.isnan(r.road_jaccard_vs_truth):
            jac_by_pipe[r.pipeline].append(r.road_jaccard_vs_truth)
        lat_by_pipe[r.pipeline].append(r.latency_median_us_per_point)

    labels = [PIPELINE_LABELS[p] for p in PIPELINES]
    colors = ["#bdbdbd", "#64b5f6", "#1565c0"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (axes[0], [by_pipe[p] for p in PIPELINES], "Compression Ratio (byte-based)", "original / encoded", False),
        (axes[1], [jac_by_pipe[p] for p in PIPELINES], "Road-segment Jaccard vs ground truth", "Jaccard", False),
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
        f"HYSOC-N evaluation - {dataset} ({len(rows)//len(PIPELINES)} trajectories)",
        fontsize=15, fontweight="bold",
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
        description="HYSOC-N end-to-end evaluation (NYC + SanFranCentre)."
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
        "--trace-gamma",
        type=float,
        default=None,
        help="Override the operating point's TRACE gamma (rare; default: use op value).",
    )
    parser.add_argument(
        "--trace-k",
        type=int,
        default=None,
        help="Override the operating point's TRACE k (rare; default: use op value).",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    op: OperatingPoint = OPERATING_POINTS[args.operating_point]
    eff_gamma = args.trace_gamma if args.trace_gamma is not None else op.trace_gamma
    eff_k = args.trace_k if args.trace_k is not None else op.trace_k

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
        f"gamma={eff_gamma} m/s, k={eff_k}, "
        f"max_files={args.max_files}\n"
    )

    trace_config = TraceConfig(
        gamma=eff_gamma,
        epsilon=op.trace_epsilon,
        k=eff_k,
        cleanup_threshold=op.trace_cleanup_threshold,
        decay_lambda=op.trace_decay_lambda,
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
            op=op,
            trace_config=trace_config,
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
        "trace_gamma_mps": eff_gamma,
        "trace_k": eff_k,
        "max_files": args.max_files,
        "latency_warmup": LATENCY_WARMUP,
        "latency_repeats": {
            "plain_trace": LATENCY_REPEATS_PLAIN_TRACE,
            "baseline_n": LATENCY_REPEATS_BASELINE_N,
            "hysoc_n": LATENCY_REPEATS_HYSOC_N,
        },
        "bytes_per_point": BYTES_PER_POINT,
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
