"""HYSOC-G cross-city sensitivity, variant 3: annotated heatmaps.

One 2x2 heatmap per metric (CR, mean SED, Stop F1) over operating point (rows)
and evaluation set (columns), annotated with median (IQR); per-city cells get a
thick black outline.
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
FIGSIZE = (11.0, 4.0)


def load_hysoc_g(run: str, eval_file: str):
    import pandas as pd

    path = os.path.join(EVAL_ROOT, run, eval_file)
    df = pd.read_csv(path)
    return df[df["pipeline"] == PIPELINE].copy()


def summary(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


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

    cells = {
        ("NYC op", "NYC eval"): nyc_op_nyc,
        ("NYC op", "SF eval"): nyc_op_sf,
        ("SF op", "NYC eval"): sf_op_nyc,
        ("SF op", "SF eval"): sf_op_sf,
    }
    diagonal = {("NYC op", "NYC eval"), ("SF op", "SF eval")}
    ops = ["NYC op", "SF op"]
    evals = ["NYC eval", "SF eval"]

    metrics = [
        ("compression_ratio", "Compression ratio", "viridis", False),
        ("mean_sed_m", "Mean SED (m)", "magma_r", False),
        ("stop_f1", "Stop $F_1$", "RdYlGn", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    for ax, (col, label, cmap_name, fixed_01) in zip(axes, metrics):
        grid = np.zeros((2, 2))
        annot = np.empty((2, 2), dtype=object)
        for i, op in enumerate(ops):
            for j, ev in enumerate(evals):
                med, q1, q3 = summary(cells[(op, ev)][col])
                grid[i, j] = med
                annot[i, j] = f"{med:.2f}\n({q1:.2f}–{q3:.2f})"

        if fixed_01:
            im = ax.imshow(grid, cmap=cmap_name, vmin=0.0, vmax=1.0, aspect="auto")
        else:
            im = ax.imshow(grid, cmap=cmap_name, aspect="auto")

        for i, op in enumerate(ops):
            for j, ev in enumerate(evals):
                value = grid[i, j]
                vmin, vmax = im.get_clim()
                norm = (value - vmin) / (vmax - vmin + 1e-12)
                color = "white" if norm < 0.45 else "black"
                if cmap_name == "RdYlGn":
                    color = "black"
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=9, color=color)
                if (op, ev) in diagonal:
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.4, zorder=5,
                    )
                    ax.add_patch(rect)

        ax.set_xticks(range(len(evals)))
        ax.set_xticklabels(evals)
        ax.set_yticks(range(len(ops)))
        ax.set_yticklabels(ops)
        ax.set_title(label)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    diag_handle = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.0, label="per-city config")
    fig.legend(handles=[diag_handle], loc="lower center", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("HYSOC-G cross-city sensitivity: median (IQR) per configuration", y=1.02)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out_path = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
