"""STEP (D, T) sweep Pareto figure: Stop F_1 vs per-point latency.

Single Stop F_1 vs latency scatter across all (D, T) configurations. Marker
shape encodes the dwell threshold T, fill colour encodes the stay-point radius
D, and the selected operating point is overlaid with a hollow red square.
"""

# ruff: noqa: E402

import argparse
import csv
import logging
import math
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from constants.operating_points import OPERATING_POINTS

from _latex_style import apply_latex_style, recommended_figsize

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SWEEP_OUTPUT_ROOT = os.path.join(
    project_root, "results", "sweeps", "module1_step_stss"
)
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

SWEEP_RUN_DIR: str | None = None

DPI = 300
FIGSIZE = recommended_figsize(1, 1, panel_aspect=0.7)

F1_FEASIBILITY_THRESHOLD = 0.6

T_MARKERS: dict[float, str] = {
    10.0: "o",
    15.0: "^",
    30.0: "s",
    60.0: "D",
}

COLUMNS = {
    "eps": "eps_m",
    "t": "t_s",
    "f1": "f1_median",
    "passes": "passes_plan_heuristic",
    "latency_ms": "step_ms_per_point_median",
}


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _passes_predicate(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() in {"1", "1.0", "True", "true"}
    return bool(value)


def resolve_sweep_run_dir(dataset: str, logger: logging.Logger) -> str:
    if SWEEP_RUN_DIR is not None:
        path = os.path.join(SWEEP_OUTPUT_ROOT, SWEEP_RUN_DIR)
        if not os.path.isdir(path):
            logger.error(f"SWEEP_RUN_DIR not found: {path}")
            sys.exit(1)
        return path
    if not os.path.isdir(SWEEP_OUTPUT_ROOT):
        logger.error(f"Sweep output root missing: {SWEEP_OUTPUT_ROOT}")
        logger.error("Run `uv run python src/sweeps/module1_step_stss.py` first.")
        sys.exit(1)
    all_dirs = [
        d for d in os.listdir(SWEEP_OUTPUT_ROOT)
        if os.path.isdir(os.path.join(SWEEP_OUTPUT_ROOT, d))
    ]
    if dataset == "sf":
        candidates = [d for d in all_dirs if d.startswith("sf_")]
    else:
        candidates = [d for d in all_dirs if not d.startswith("sf_")]
    if not candidates:
        logger.error(f"No run folders matching dataset={dataset!r} under {SWEEP_OUTPUT_ROOT}")
        sys.exit(1)
    return os.path.join(SWEEP_OUTPUT_ROOT, max(candidates))


def load_aggregated(run_dir: str, logger: logging.Logger) -> list[dict]:
    csv_path = os.path.join(run_dir, "aggregated.csv")
    if not os.path.exists(csv_path):
        logger.error(f"aggregated.csv not found in {run_dir}")
        sys.exit(1)
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Loaded {len(rows)} configurations from {csv_path}")
    return rows


def _parse_points(rows: list[dict], logger: logging.Logger) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        eps = _to_float(r[COLUMNS["eps"]])
        t = _to_float(r[COLUMNS["t"]])
        f1 = _to_float(r[COLUMNS["f1"]])
        ms = _to_float(r[COLUMNS["latency_ms"]])
        passes = _passes_predicate(r[COLUMNS["passes"]])
        if any(math.isnan(v) for v in (eps, t, f1, ms)):
            logger.warning("Skipping row with non-numeric fields: %s", r)
            continue
        out.append({
            "eps": eps,
            "t": t,
            "f1": f1,
            "latency_us": ms * 1000.0,
            "passes": passes,
        })
    return out


def main() -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    apply_latex_style()

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(OPERATING_POINTS),
        default="nyc",
        help="Calibration dataset to plot (default: nyc).",
    )
    args = parser.parse_args()

    op = OPERATING_POINTS[args.dataset]
    selected_eps = op.stop_max_eps_m
    selected_t = op.stop_min_duration_s

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    run_dir = resolve_sweep_run_dir(args.dataset, logger)
    rows = load_aggregated(run_dir, logger)
    points = _parse_points(rows, logger)
    if not points:
        logger.error("No valid rows in aggregated.csv")
        sys.exit(1)

    eps_values = sorted({p["eps"] for p in points})
    t_values_present = sorted({p["t"] for p in points})
    norm = Normalize(vmin=min(eps_values), vmax=max(eps_values))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(
        F1_FEASIBILITY_THRESHOLD,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    for p in points:
        marker = T_MARKERS.get(p["t"], "o")
        color = cmap(norm(p["eps"]))
        edge = "black" if p["passes"] else "white"
        edge_w = 1.2 if p["passes"] else 0.5
        ax.scatter(
            p["latency_us"],
            p["f1"],
            marker=marker,
            s=110,
            facecolor=color,
            edgecolor=edge,
            linewidth=edge_w,
            zorder=3,
            alpha=0.95,
        )

    sel = next(
        (p for p in points
         if math.isclose(p["eps"], selected_eps, rel_tol=1e-9)
         and math.isclose(p["t"], selected_t, rel_tol=1e-9)),
        None,
    )
    if sel is None:
        logger.warning("Selected operating point not found in sweep rows")
    else:
        ax.scatter(
            sel["latency_us"],
            sel["f1"],
            marker="s",
            s=320,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=2.0,
            zorder=4,
        )

    ax.set_xlabel(r"Per-point latency ($\mu$s)")
    ax.set_ylabel("Stop $F_1$ (median)")
    ax.grid(True, alpha=0.3)

    floor_handle = mlines.Line2D(
        [], [], color="#888888", linestyle="--", linewidth=1.0,
        label=f"$F_1$ = {F1_FEASIBILITY_THRESHOLD:g} (feasibility floor)",
    )
    feasible_handle = mlines.Line2D(
        [], [], marker="o", color="white", markerfacecolor="#cccccc",
        markeredgecolor="black", markeredgewidth=1.2, markersize=9,
        linestyle="None", label="passes plan heuristic",
    )
    selected_handle = mlines.Line2D(
        [], [], marker="s", color="white", markerfacecolor="none",
        markeredgecolor="#d62728", markeredgewidth=2.0, markersize=12,
        linestyle="None",
        label=(
            f"selected ($D = {int(selected_eps)}$ m, "
            f"$T = {int(selected_t)}$ s)"
        ),
    )
    t_handles = [
        mlines.Line2D(
            [], [], marker=T_MARKERS[t], color="white",
            markerfacecolor="#888888", markeredgecolor="#444444",
            markersize=9, linestyle="None", label=f"$T = {int(t)}$ s",
        )
        for t in t_values_present if t in T_MARKERS
    ]

    legend_handles = [floor_handle, feasible_handle, selected_handle, *t_handles]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=8,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label(r"Stay-point radius $D$ (m)")
    cbar.set_ticks(eps_values)
    cbar.ax.set_yticklabels([f"{int(v)}" for v in eps_values])

    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    suffix = "" if args.dataset == "nyc" else f"_{args.dataset}"
    run_out_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, f"{timestamp}{suffix}")
    os.makedirs(run_out_dir, exist_ok=True)
    out_path = os.path.join(run_out_dir, f"{SCRIPT_NAME}{suffix}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved figure: {out_path}")
    logger.info(f"Sourced from sweep run: {os.path.basename(run_dir)}")


if __name__ == "__main__":
    main()
