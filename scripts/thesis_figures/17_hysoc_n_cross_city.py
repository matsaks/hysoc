"""HYSOC-N cross-city sensitivity slope figure (mirrors 16 variant 2).

Three panels (CR, road Jaccard, Stop F1) over a discrete [NYC eval, SF eval]
x-axis, with one median trajectory per operating point and the matched-op floor
marked on the F1 panel.
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
EVAL_ROOT = os.path.join(project_root, "results", "experiments", "hysoc_n_eval")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

NYC_OP_RUN = "0527_1649_op-nyc"
SF_OP_RUN = "0527_1728_op-sf_centre"
PIPELINE = "hysoc_n"

DPI = 300
FIGSIZE = (11.0, 4.2)

OP_COLORS = {"NYC op": "#1f77b4", "SF op": "#d62728"}


def load_hysoc_n(run: str, eval_file: str):
    import pandas as pd

    path = os.path.join(EVAL_ROOT, run, eval_file)
    df = pd.read_csv(path)
    return df[df["pipeline"] == PIPELINE].copy()


def median_finite(values) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr))


def mean_finite(values) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr))


def main() -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)
    apply_latex_style(use_latex=False)

    nyc_op_nyc = load_hysoc_n(NYC_OP_RUN, "nyc_per_trajectory.csv")
    nyc_op_sf = load_hysoc_n(NYC_OP_RUN, "sf_centre_per_trajectory.csv")
    sf_op_nyc = load_hysoc_n(SF_OP_RUN, "nyc_per_trajectory.csv")
    sf_op_sf = load_hysoc_n(SF_OP_RUN, "sf_centre_per_trajectory.csv")

    series = {
        ("NYC op", "NYC eval"): nyc_op_nyc,
        ("NYC op", "SF eval"): nyc_op_sf,
        ("SF op", "NYC eval"): sf_op_nyc,
        ("SF op", "SF eval"): sf_op_sf,
    }
    diagonal = {("NYC op", "NYC eval"), ("SF op", "SF eval")}

    metrics = [
        ("compression_ratio", "Compression ratio", None, median_finite),
        ("road_jaccard_vs_truth", "Road Jaccard", "jaccard", mean_finite),
        ("stop_f1", "Stop $F_1$", "f1", median_finite),
    ]
    eval_sets = ["NYC eval", "SF eval"]
    ops = ["NYC op", "SF op"]
    x_pos = {"NYC eval": 0, "SF eval": 1}

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    for ax, (col, label, panel_kind, agg) in zip(axes, metrics):
        medians_by_op = {}
        for op in ops:
            xs, meds = [], []
            for ev in eval_sets:
                m = agg(series[(op, ev)][col])
                xs.append(x_pos[ev]); meds.append(m)  # noqa: E702
            medians_by_op[op] = meds
            ax.plot(xs, meds, color=OP_COLORS[op], linewidth=1.8, zorder=2, label=op)
            for ev, m in zip(eval_sets, meds):
                is_diag = (op, ev) in diagonal
                ax.scatter(
                    x_pos[ev], m,
                    s=140 if is_diag else 80,
                    facecolor=OP_COLORS[op] if is_diag else "white",
                    edgecolor=OP_COLORS[op],
                    linewidth=1.8,
                    marker="o" if is_diag else "s",
                    zorder=4,
                )
                xoff = -16 if x_pos[ev] == 0 else 16
                yoff = 0
                if panel_kind == "jaccard":
                    yoff = 10 if op == "NYC op" else -10
                ax.annotate(
                    f"{m:.2f}",
                    xy=(x_pos[ev], m),
                    xytext=(xoff, yoff),
                    textcoords="offset points",
                    fontsize=11,
                    ha="right" if x_pos[ev] == 0 else "left",
                    va="center",
                    color=OP_COLORS[op],
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95),
                    zorder=5,
                )

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(list(x_pos.keys()))
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, zorder=0)

        if panel_kind is None:
            top = max(max(medians_by_op["NYC op"]), max(medians_by_op["SF op"])) * 1.15
            ax.set_ylim(0, top)

        if panel_kind == "jaccard":
            ax.set_ylim(0.90, 1.02)

        if panel_kind == "f1":
            ax.set_ylim(0.0, 1.0)
            matched_floor = 0.67
            ax.axhline(matched_floor, color="#444444", linestyle=":", linewidth=0.9, zorder=1)
            ax.text(
                0.5, 0.94,
                f"matched-op floor: $F_1 = {matched_floor:.2f}$",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
            )

    op_handles = [
        mlines.Line2D([], [], color=OP_COLORS["NYC op"], linewidth=2.0, label="NYC op"),
        mlines.Line2D([], [], color=OP_COLORS["SF op"], linewidth=2.0, label="SF op"),
        mlines.Line2D([], [], color="#444444", marker="o", linestyle="None",
                      markersize=9, markerfacecolor="#444444", label="matched (per-city)"),
        mlines.Line2D([], [], color="#444444", marker="s", linestyle="None",
                      markersize=8, markerfacecolor="white", markeredgewidth=1.5, label="cross-city"),
    ]
    fig.legend(
        handles=op_handles, loc="lower center", ncol=4,
        frameon=True, fontsize=10.5, bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle("HYSOC-N cross-city sensitivity: matched op wins Stop $F_1$; road Jaccard is invariant", y=1.02)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out_path = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
