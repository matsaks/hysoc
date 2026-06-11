"""HYSOC-G SQUISH+DP sweep Pareto figure: move CR vs mean SED.

Single move-CR-vs-mean-SED scatter across a subset of the (beta, epsilon) grid.
Points within each epsilon are connected in increasing beta order, the pure-DP
control (beta = 0) is drawn with a diamond, and the selected operating point is
overlaid with a hollow red square.
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
    project_root, "results", "sweeps", "module3_hysoc_g_move"
)
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

SWEEP_RUN_DIR: str | None = None

DPI = 300
FIGSIZE = recommended_figsize(1, 1, panel_aspect=0.7)

SQUISH_DP_STRATEGY = "squish_dp"
PURE_DP_STRATEGY = "pure_dp"

EPS_SUBSET_METERS: tuple[float, ...] = (5.0, 6.0, 7.0, 10.0, 20.0, 50.0)
SATURATION_BETA_THRESHOLD: int = 500

COLUMNS = {
    "strategy": "strategy",
    "capacity": "capacity",
    "eps": "dp_epsilon_m",
    "cr": "move_cr_median",
    "mean_sed": "move_mean_sed_m_median",
}


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def resolve_sweep_run_dir(dataset: str, logger: logging.Logger) -> str:
    if SWEEP_RUN_DIR is not None:
        path = os.path.join(SWEEP_OUTPUT_ROOT, SWEEP_RUN_DIR)
        if not os.path.isdir(path):
            logger.error(f"SWEEP_RUN_DIR not found: {path}")
            sys.exit(1)
        return path
    if not os.path.isdir(SWEEP_OUTPUT_ROOT):
        logger.error(f"Sweep output root missing: {SWEEP_OUTPUT_ROOT}")
        logger.error("Run `uv run python src/sweeps/module3_hysoc_g_move.py` first.")
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


def _eps_in_subset(eps: float) -> bool:
    return any(math.isclose(eps, target, rel_tol=1e-9, abs_tol=1e-9)
               for target in EPS_SUBSET_METERS)


def _parse_curves(rows: list[dict], logger: logging.Logger) -> dict[float, list[dict]]:
    curves: dict[float, list[dict]] = {}
    for r in rows:
        strategy = r[COLUMNS["strategy"]]
        if strategy not in {SQUISH_DP_STRATEGY, PURE_DP_STRATEGY}:
            continue
        eps = _to_float(r[COLUMNS["eps"]])
        beta = _to_float(r[COLUMNS["capacity"]])
        cr = _to_float(r[COLUMNS["cr"]])
        sed = _to_float(r[COLUMNS["mean_sed"]])
        if any(math.isnan(v) for v in (eps, beta, cr, sed)):
            continue
        if not _eps_in_subset(eps):
            continue
        curves.setdefault(eps, []).append({
            "strategy": strategy,
            "beta": beta,
            "cr": cr,
            "sed": sed,
        })

    for eps, pts in curves.items():
        pts.sort(key=lambda p: p["beta"])
        if not any(p["strategy"] == PURE_DP_STRATEGY for p in pts):
            logger.warning("No pure_dp row at eps=%s m; line will start at smallest beta", eps)

    return curves


def main() -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

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
    sel_eps_target = op.dp_epsilon_m
    sel_beta_target = float(op.squish_capacity)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    run_dir = resolve_sweep_run_dir(args.dataset, logger)
    rows = load_aggregated(run_dir, logger)
    curves = _parse_curves(rows, logger)
    if not curves:
        logger.error("No valid (beta, epsilon) rows for the chosen epsilon subset")
        sys.exit(1)

    eps_sorted = sorted(curves.keys())
    norm = LogNorm(vmin=min(eps_sorted), vmax=max(eps_sorted))
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for eps in eps_sorted:
        pts = curves[eps]
        color = cmap(norm(eps))
        xs = [p["sed"] for p in pts]
        ys = [p["cr"] for p in pts]

        ax.plot(
            xs, ys,
            "-",
            color=color,
            linewidth=1.3,
            alpha=0.55,
            zorder=2,
        )

        for p in pts:
            is_pure = p["strategy"] == PURE_DP_STRATEGY
            is_saturated = p["beta"] >= SATURATION_BETA_THRESHOLD
            if is_pure:
                marker = "D"
                size = 80
                edge = "black"
                edge_w = 0.8
                face = color
                alpha = 0.95
            elif is_saturated:
                marker = "o"
                size = 36
                edge = color
                edge_w = 1.0
                face = "white"
                alpha = 0.95
            else:
                marker = "o"
                size = 70
                edge = "white"
                edge_w = 0.6
                face = color
                alpha = 0.95
            ax.scatter(
                p["sed"], p["cr"],
                marker=marker,
                s=size,
                facecolor=face,
                edgecolor=edge,
                linewidth=edge_w,
                zorder=3,
                alpha=alpha,
            )

    sel_eps = sel_eps_target
    sel_beta = sel_beta_target
    sel_pt = None
    if sel_eps in curves:
        for p in curves[sel_eps]:
            if math.isclose(p["beta"], sel_beta, rel_tol=1e-9, abs_tol=1e-9) \
               and p["strategy"] == SQUISH_DP_STRATEGY:
                sel_pt = p
                break
    if sel_pt is None:
        logger.warning(
            "Selected (beta=%s, eps=%s) not found in curve set",
            sel_beta_target, sel_eps_target,
        )
    else:
        ax.scatter(
            sel_pt["sed"], sel_pt["cr"],
            marker="s",
            s=320,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=2.0,
            zorder=5,
        )

    ax.set_xlabel("Mean SED on move segments (m, median)")
    ax.set_ylabel("Move CR (median)")
    ax.grid(True, alpha=0.3)

    pure_handle = mlines.Line2D(
        [], [], marker="D", color="white", markerfacecolor="#888888",
        markeredgecolor="black", markeredgewidth=0.8, markersize=8,
        linestyle="None", label=r"$\beta = 0$ (pure DP)",
    )
    transient_handle = mlines.Line2D(
        [], [], marker="o", color="white", markerfacecolor="#888888",
        markeredgecolor="white", markeredgewidth=0.6, markersize=8,
        linestyle="None", label=r"$\beta \in \{50, 100, 200\}$ (eviction active)",
    )
    saturated_handle = mlines.Line2D(
        [], [], marker="o", color="white", markerfacecolor="white",
        markeredgecolor="#888888", markeredgewidth=1.0, markersize=7,
        linestyle="None",
        label=fr"$\beta \geq {SATURATION_BETA_THRESHOLD}$ (saturated, coincides with pure DP)",
    )
    selected_handle = mlines.Line2D(
        [], [], marker="s", color="white", markerfacecolor="none",
        markeredgecolor="#d62728", markeredgewidth=2.0, markersize=12,
        linestyle="None",
        label=(
            fr"selected ($\beta = {int(sel_beta_target)}$, "
            fr"$\varepsilon = {sel_eps_target:g}$ m)"
        ),
    )

    legend_handles = [pure_handle, transient_handle, saturated_handle]
    if sel_pt is not None:
        legend_handles.append(selected_handle)
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
    cbar.set_label(r"DP tolerance $\varepsilon$ (m, log scale)")
    cbar.set_ticks(eps_sorted)
    cbar.ax.set_yticklabels([f"{v:g}" for v in eps_sorted])
    cbar.ax.minorticks_off()

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
