"""
Aggregate per-trajectory sweep results into per-config comparison rows.

Input:
  data/processed/param_sweep_step_stss.csv

Output:
  data/processed/param_sweep_step_stss_aggregated.csv

Usage:
  uv run python scripts/demo_28_aggregate_param_sweep.py
  uv run python scripts/demo_28_aggregate_param_sweep.py --sort-by best
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).parent

DEFAULT_IN = project_root / "data" / "processed" / "param_sweep_step_stss.csv"
DEFAULT_OUT = (
    project_root / "data" / "processed" / "param_sweep_step_stss_aggregated.csv"
)

GROUP_COLS = ["eps_m", "t_s", "min_samples"]
METRIC_COLS = [
    "n_points",
    "step_n_stops",
    "step_n_moves",
    "step_mean_stop_s",
    "step_mean_move_pts",
    "step_wall_s",
    "stss_n_stops",
    "stss_n_moves",
    "stss_mean_stop_s",
    "stss_mean_move_pts",
    "stss_wall_s",
    "f1",
    "precision",
    "recall",
    "matched_iou_mean",
]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    mean_val = statistics.mean(values)
    med_val = statistics.median(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_median": med_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    by_cfg: dict[tuple[float, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (float(row["eps_m"]), float(row["t_s"]), int(float(row["min_samples"])))
        by_cfg[key].append(row)

    aggregated: list[dict[str, float | int | str]] = []
    for (eps_m, t_s, min_samples), cfg_rows in sorted(by_cfg.items()):
        out: dict[str, float | int | str] = {
            "eps_m": eps_m,
            "t_s": t_s,
            "min_samples": min_samples,
            "n_trajectories": len(cfg_rows),
        }

        for metric in METRIC_COLS:
            values = [as_float(r, metric) for r in cfg_rows]
            out.update(summarize(values, metric))

        # Helper columns for quickly selecting "good" configs.
        out["passes_plan_heuristic"] = int(
            out["step_n_stops_median"] >= 2
            and out["step_n_moves_median"] >= 2
            and out["f1_median"] > 0.6
        )

        # Convenience throughput columns.
        out["step_ms_per_point_mean"] = (
            out["step_wall_s_mean"] / out["n_points_mean"] * 1000.0
            if out["n_points_mean"] > 0
            else 0.0
        )
        out["step_ms_per_point_median"] = (
            out["step_wall_s_median"] / out["n_points_median"] * 1000.0
            if out["n_points_median"] > 0
            else 0.0
        )

        aggregated.append(out)

    return aggregated


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_fields(
    rows: list[dict[str, float | int | str]], fieldnames: list[str]
) -> list[dict[str, float | int | str]]:
    return [{k: row[k] for k in fieldnames} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-trajectory STEP/STSS sweep results."
    )
    parser.add_argument("--in", dest="in_csv", default=str(DEFAULT_IN))
    parser.add_argument("--out", dest="out_csv", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--sort-by",
        choices=["grid", "best"],
        default="best",
        help="grid=eps/t order, best=heuristic then F1.",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Write only the thesis-comparison subset of columns.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    aggregated = aggregate_rows(rows)
    if args.sort_by == "best":
        aggregated.sort(
            key=lambda r: (
                -int(r["passes_plan_heuristic"]),
                -float(r["f1_median"]),
                float(r["eps_m"]),
                float(r["t_s"]),
            )
        )

    if args.slim:
        slim_fields = [
            "eps_m",
            "t_s",
            "step_n_stops_mean",
            "step_n_stops_median",
            "step_n_stops_std",
            "step_n_moves_mean",
            "stss_n_stops_mean",
            "stss_n_stops_median",
            "stss_n_stops_std",
            "stss_n_moves_mean",
            "f1_mean",
            "f1_median",
            "f1_std",
            "f1_min",
            "f1_max",
            "precision_mean",
            "precision_median",
            "precision_std",
            "precision_min",
            "precision_max",
            "recall_mean",
            "recall_median",
            "recall_std",
            "recall_min",
            "recall_max",
            "matched_iou_mean_mean",
            "matched_iou_mean_median",
            "matched_iou_mean_std",
            "matched_iou_mean_min",
            "matched_iou_mean_max",
            "passes_plan_heuristic",
            "step_ms_per_point_mean",
            "step_ms_per_point_median",
        ]
        aggregated = select_fields(aggregated, slim_fields)

    write_csv(aggregated, out_path)
    print(f"Read {len(rows)} rows from {in_path}")
    print(f"Wrote {len(aggregated)} aggregated config rows to {out_path}")


if __name__ == "__main__":
    main()
