"""HYSOC-G cross-city sensitivity, variant 1: violin grid.

3x2 grid (rows: CR, mean SED, Stop F1; columns: NYC, SF eval) with two violins
per panel comparing the two operating points; per-city configs are shaded.
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
FIGSIZE = (8.5, 8.0)

OP_COLORS = {"NYC op": "#1f77b4", "SF op": "#d62728"}
DIAGONAL_FACE = "#f3f3f3"


def load_hysoc_g(run: str, eval_file: str):
    import pandas as pd

    path = os.path.join(EVAL_ROOT, run, eval_file)
    df = pd.read_csv(path)
    return df[df["pipeline"] == PIPELINE].copy()


def main() -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)
    apply_latex_style(use_latex=False)

    nyc_op_nyc = load_hysoc_g(NYC_OP_RUN, "nyc_per_trajectory.csv")
    nyc_op_sf = load_hysoc_g(NYC_OP_RUN, "sf_centre_per_trajectory.csv")
    sf_op_nyc = load_hysoc_g(SF_OP_RUN, "nyc_per_trajectory.csv")
    sf_op_sf = load_hysoc_g(SF_OP_RUN, "sf_centre_per_trajectory.csv")

    panels = {
        ("NYC eval", "NYC op"): nyc_op_nyc,
        ("NYC eval", "SF op"): sf_op_nyc,
        ("SF eval", "NYC op"): nyc_op_sf,
        ("SF eval", "SF op"): sf_op_sf,
    }
    diagonal = {("NYC eval", "NYC op"), ("SF eval", "SF op")}

    metrics = [
        ("compression_ratio", "Compression ratio", (0, 60)),
        ("mean_sed_m", "Mean SED (m)", (0, 60)),
        ("stop_f1", "Stop $F_1$", (-0.02, 1.02)),
    ]
    eval_cols = ["NYC eval", "SF eval"]
    ops = ["NYC op", "SF op"]

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, sharey="row")

    for r, (col, label, ylim) in enumerate(metrics):
        for c, eval_name in enumerate(eval_cols):
            ax = axes[r, c]
            data = []
            for op in ops:
                series = panels[(eval_name, op)][col].dropna().to_numpy()
                series = series[np.isfinite(series)]
                data.append(series)

            parts = ax.violinplot(
                data,
                positions=[0, 1],
                widths=0.8,
                showmedians=False,
                showextrema=False,
            )
            for body, op in zip(parts["bodies"], ops):
                body.set_facecolor(OP_COLORS[op])
                body.set_alpha(0.55)
                body.set_edgecolor("#222222")
                body.set_linewidth(0.7)

            for i, (series, op) in enumerate(zip(data, ops)):
                if len(series) == 0:
                    continue
                med = float(np.median(series))
                q1, q3 = np.percentile(series, [25, 75])
                if (eval_name, op) in diagonal:
                    ax.axvspan(i - 0.5, i + 0.5, color=DIAGONAL_FACE, zorder=0)
                ax.hlines(med, i - 0.2, i + 0.2, color="black", linewidth=1.6, zorder=3)
                ax.vlines(i, q1, q3, color="black", linewidth=1.2, zorder=3)
                ax.text(
                    i, 0.97, f"{med:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(ops if r == 2 else [])
            ax.set_ylim(*ylim)
            ax.grid(True, axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel(label)
            if r == 0:
                ax.set_title(eval_name)

    diag_patch = mpatches.Patch(facecolor=DIAGONAL_FACE, edgecolor="#888888", label="per-city config")
    fig.legend(handles=[diag_patch], loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("HYSOC-G cross-city sensitivity: per-trajectory distributions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out_path = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
