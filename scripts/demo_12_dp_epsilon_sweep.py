import os
import sys
import csv
import json
import argparse
from datetime import datetime
import logging
import time
from typing import Dict, List

import numpy as np

# Add project root to sys.path to find packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from core.point import Point
from eval import calculate_sed_stats
from engines.dp import DouglasPeuckerCompressor


def load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    points: List[Point] = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")

    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: time, latitude, longitude (see existing demo loaders)
            try:
                dt = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue

            points.append(Point(lat=lat, lon=lon, timestamp=dt, obj_id=obj_id))

    return points


def pooled_sed_metrics(sed_errors: List[float]) -> Dict[str, float]:
    if not sed_errors:
        return {
            "average_sed": 0.0,
            "max_sed": 0.0,
            "rmse": 0.0,
            "p95_sed": 0.0,
        }

    arr = np.asarray(sed_errors, dtype=float)
    return {
        "average_sed": float(np.mean(arr)),
        "max_sed": float(np.max(arr)),
        "rmse": float(np.sqrt(np.mean(np.square(arr)))),
        "p95_sed": float(np.percentile(arr, 95)),
    }


def evaluate_epsilon(
    epsilon_meters: float,
    trajectories_by_file: Dict[str, List[Point]],
    logger: logging.Logger,
    log_every: int,
) -> Dict[str, float]:
    compressor = DouglasPeuckerCompressor(epsilon_meters=epsilon_meters)

    sed_errors_pooled: List[float] = []
    total_original = 0
    total_compressed = 0
    used_files = 0

    t_start = time.perf_counter()
    total_files = len(trajectories_by_file)

    for file_idx, (_file, original_points) in enumerate(trajectories_by_file.items(), start=1):
        if len(original_points) < 2:
            logger.debug(
                f"epsilon={epsilon_meters}: skipping file #{file_idx}/{total_files} (n_points={len(original_points)})"
            )
            continue

        compressed_points = compressor.compress(original_points)

        total_original += len(original_points)
        total_compressed += len(compressed_points)

        sed_stats = calculate_sed_stats(original_points, compressed_points)
        sed_errors = sed_stats.get("sed_errors", [])
        if sed_errors:
            sed_errors_pooled.extend(sed_errors)

        used_files += 1

        if log_every > 0 and (file_idx % log_every == 0 or file_idx == total_files):
            partial_cr = (total_original / total_compressed) if total_compressed > 0 else 1.0
            logger.info(
                f"epsilon={epsilon_meters}: processed {file_idx}/{total_files} files "
                f"(used={used_files}); partial_cr={partial_cr:.3f}; "
                f"last_avg_sed={sed_stats.get('average_sed', 0.0):.3f} m"
            )

    # Global pooled SED metrics
    metrics = pooled_sed_metrics(sed_errors_pooled)

    # Global CR using pooled point counts
    cr = (total_original / total_compressed) if total_compressed > 0 else 1.0
    metrics["cr"] = float(cr)
    metrics["total_original_points"] = float(total_original)
    metrics["total_compressed_points"] = float(total_compressed)
    metrics["used_files"] = float(used_files)
    metrics["runtime_s"] = float(time.perf_counter() - t_start)

    return metrics


def parse_epsilon_grid(grid_str: str) -> List[float]:
    # "1,2,5,10" -> [1.0, 2.0, 5.0, 10.0]
    parts = [p.strip() for p in grid_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty epsilon grid.")
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline geometric benchmark: sweep DP epsilon and optimize CR under a pooled p95 SED constraint."
    )
    parser.add_argument("--subset", default="subset_50", help="Folder under data/raw/ (default: subset_50)")
    parser.add_argument(
        "--epsilon-grid",
        default="0.5,1,2,3,5,7,10,12,15,18,20,25,30",
        help="Comma-separated epsilon_meters values.",
    )
    parser.add_argument("--baseline-epsilon", type=float, default=15.0, help="Baseline epsilon used to calibrate p95 threshold.")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING (default: INFO)")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N files during evaluation (default: 10). Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "processed", "demo_12_dp_epsilon_sweep"),
        help="Output root directory (default: data/processed/demo_12_dp_epsilon_sweep).",
    )
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid --log-level: {args.log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("dp_epsilon_sweep")

    subset_dir = os.path.join(project_root, "data", "raw", args.subset)
    pattern = os.path.join(subset_dir, "*.csv")

    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"subset directory not found: {subset_dir}")

    # Discover all 50 files
    csv_files = sorted([f for f in (os.listdir(subset_dir)) if f.lower().endswith(".csv")])
    if len(csv_files) != 50:
        print(f"Warning: expected 50 csv files in {subset_dir}, found {len(csv_files)}.")

    trajectories_by_file: Dict[str, List[Point]] = {}
    total_loaded_points = 0
    for file_idx, filename in enumerate(csv_files, start=1):
        filepath = os.path.join(subset_dir, filename)
        obj_id = os.path.splitext(filename)[0]
        points = load_trajectory(filepath, obj_id=obj_id)
        trajectories_by_file[filename] = points
        total_loaded_points += len(points)

        if args.log_every > 0 and (file_idx % args.log_every == 0 or file_idx == len(csv_files)):
            logger.info(
                f"Loaded {file_idx}/{len(csv_files)} files; total_loaded_points={total_loaded_points}"
            )

    epsilon_grid = parse_epsilon_grid(args.epsilon_grid)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, args.output_dir, f"{timestamp}_{args.subset}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Baseline threshold calibration
    logger.info(
        f"Calibrating p95 threshold using baseline epsilon={args.baseline_epsilon} on all files "
        f"(n_files={len(trajectories_by_file)})..."
    )
    baseline_metrics = evaluate_epsilon(
        args.baseline_epsilon,
        trajectories_by_file,
        logger=logger,
        log_every=args.log_every,
    )
    threshold_p95 = baseline_metrics["p95_sed"]
    logger.info(f"Baseline p95(SED) = {threshold_p95:.6f} m (runtime_s={baseline_metrics.get('runtime_s', 0.0):.2f})")

    # 2) Sweep epsilons
    logger.info(f"Sweeping epsilons: {epsilon_grid}")
    rows: List[Dict[str, float]] = []
    for eps in epsilon_grid:
        logger.info(f"Evaluating epsilon_meters={eps} ...")
        metrics = evaluate_epsilon(
            eps,
            trajectories_by_file,
            logger=logger,
            log_every=args.log_every,
        )
        logger.info(
            f"epsilon={eps}: cr={metrics['cr']:.4f}, avg_sed={metrics['average_sed']:.4f} m, "
            f"p95_sed={metrics['p95_sed']:.4f} m, runtime_s={metrics.get('runtime_s', 0.0):.2f} s"
        )
        rows.append({"epsilon_meters": float(eps), **metrics})

    # 3) Select best epsilon under constraint
    candidates = [r for r in rows if r["p95_sed"] <= threshold_p95]
    if not candidates:
        logger.warning("No epsilon met the p95 constraint; selecting best CR regardless.")
        best = max(rows, key=lambda r: (r["cr"], -r["p95_sed"], -r["average_sed"]))
    else:
        # Max CR; tie-break: smaller p95 then smaller mean SED
        best = sorted(candidates, key=lambda r: (-r["cr"], r["p95_sed"], r["average_sed"]))[0]
        logger.info(f"Selected best epsilon under constraint: {best['epsilon_meters']} (cr={best['cr']:.4f}, p95={best['p95_sed']:.4f} m)")

    # 4) Write outputs
    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = [
        "epsilon_meters",
        "cr",
        "average_sed",
        "p95_sed",
        "max_sed",
        "rmse",
        "total_original_points",
        "total_compressed_points",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    best_report = {
        "subset": args.subset,
        "epsilon_grid": epsilon_grid,
        "baseline_epsilon": args.baseline_epsilon,
        "baseline_p95_sed_m": threshold_p95,
        "selected_best": best,
        "num_candidates_under_constraint": len(candidates),
        "timestamp": timestamp,
    }

    report_path = os.path.join(out_dir, "best_epsilon.json")
    with open(report_path, "w", newline="") as f:
        json.dump(best_report, f, indent=2)

    logger.info(f"Done. Results written to: {csv_path}")
    logger.info(f"Best epsilon report written to: {report_path}")


if __name__ == "__main__":
    main()

