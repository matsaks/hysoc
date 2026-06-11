"""HYSOC-G cross-city sensitivity, variant 5: per-trajectory scatter matrix.

2x2 scatter grid (rows: operating point; columns: evaluation set) with one point
per trajectory (x: mean SED, y: CR, colour: Stop F1). Per-city panels carry a
thicker border and median crosshairs are overlaid.
"""

# ruff: noqa: E402

import logging
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from _latex_style import apply_latex_style

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
EVAL_ROOT = os.path.join(project_root, "results", "experiments", "hysoc_g_eval")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

NYC_OP_RUN = "0515_1059"
SF_OP_RUN = "0525_1604_op-sf_centre"
PIPELINE = "hysoc_g"

DPI = 300
FIGSIZE = (9.0, 8.5)


def load_hysoc_g(run: str, eval_file: str):
    import pandas as pd

    path = os.path.join(EVAL_ROOT, run, eval_file)
    df = pd.read_csv(path)
    return df[df["pipeline"] == PIPELINE].copy()


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)
    apply_latex_style(use_latex=False)

    nyc_op_nyc = load_hysoc_g(NYC_OP_RUN, "nyc_per_trajectory.csv")
    nyc_op_sf = load_hysoc_g(NYC_OP_RUN, "sf_centre_per_trajectory.csv")
    sf_op_nyc = load_hysoc_g(SF_OP_RUN, "nyc_per_trajectory.csv")
    sf_op_sf = load_hysoc_g(SF_OP_RUN, "sf_centre_per_trajectory.csv")

    cells = {
        ("NYC op", "NYC eval"): nyc_op_nyc,
        ("NYC op", "SF eval"): nyc_op_sf,
        ("SF op", "NYC eval"): sf_op_nyc,
        ("SF op", "SF eval"): sf_op_sf,
    }
    diagonal = {("NYC op", "NYC eval"), ("SF op", "SF eval")}
    ops = ["NYC op", "SF op"]
    evals = ["NYC eval", "SF eval"]

    sed_max = 60.0
    cr_max = 80.0

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)

    last_scatter = None
    for i, op in enumerate(ops):
        for j, ev in enumerate(evals):
            ax = axes[i, j]
            df = cells[(op, ev)]
            x = df["mean_sed_m"].to_numpy(dtype=float)
            y = df["compression_ratio"].to_numpy(dtype=float)
            f1 = df["stop_f1"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            sc = ax.scatter(
                x[mask], y[mask],
                c=np.where(np.isfinite(f1[mask]), f1[mask], 0.0),
                cmap="RdYlGn",
                vmin=0.0, vmax=1.0,
                s=14, alpha=0.55, linewidths=0,
            )
            last_scatter = sc

            xs_finite = x[mask]; ys_finite = y[mask]  # noqa: E702
            if len(xs_finite) > 0:
                med_x = float(np.median(xs_finite))
                med_y = float(np.median(ys_finite))
                ax.axvline(med_x, color="#222222", linewidth=0.8, alpha=0.7, zorder=2)
                ax.axhline(med_y, color="#222222", linewidth=0.8, alpha=0.7, zorder=2)
                ax.text(
                    0.97, 0.97,
                    f"med SED = {med_x:.2f} m\nmed CR = {med_y:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.85),
                )

            ax.set_xlim(0, sed_max)
            ax.set_ylim(0, cr_max)
            ax.grid(True, alpha=0.3)
            if (op, ev) in diagonal:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.2)
                    spine.set_edgecolor("black")

            if i == 0:
                ax.set_title(ev)
            if j == 0:
                ax.set_ylabel(f"{op}\nCompression ratio")
            if i == 1:
                ax.set_xlabel("Mean SED (m)")

    cbar = fig.colorbar(last_scatter, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Stop $F_1$")
    fig.suptitle("HYSOC-G cross-city sensitivity: per-trajectory (CR vs mean SED), coloured by Stop $F_1$", y=1.0)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out_path = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
