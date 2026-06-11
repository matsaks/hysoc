"""HYSOC-G cross-city sensitivity, variant 4: distributional ECDF companion.

3x2 ECDF grid (rows: CR, mean SED, Stop F1; columns: NYC, SF eval) with one ECDF
per operating point (matched solid, cross-city dashed). The mean-SED row marks
the per-city stay-point radius D and the SED-budget violation fraction.
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

# Per-city SED budgets equal to the stay-point radius D from Table 7.4.
SED_BUDGET = {"NYC eval": 15.0, "SF eval": 10.0}

DPI = 300
FIGSIZE = (9.0, 9.0)

OP_COLORS = {"NYC op": "#1f77b4", "SF op": "#d62728"}
MATCHED_EVAL = {"NYC op": "NYC eval", "SF op": "SF eval"}
BUDGET_FACE = "#d62728"


def load_hysoc_g(run: str, eval_file: str):
    import pandas as pd

    path = os.path.join(EVAL_ROOT, run, eval_file)
    df = pd.read_csv(path)
    return df[df["pipeline"] == PIPELINE].copy()


def ecdf(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    if len(arr) == 0:
        return arr, arr
    ys = np.arange(1, len(arr) + 1) / len(arr)
    return arr, ys


def fraction_above(values, threshold: float) -> tuple[float, int]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), 0
    return float((arr > threshold).mean()), int(len(arr))


def main() -> None:
    import matplotlib.lines as mlines
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

    metrics = [
        ("compression_ratio", "Compression ratio", (0, 60), None),
        ("mean_sed_m", "Mean SED (m)", (0, 50), "sed"),
        ("stop_f1", "Stop $F_1$", (0, 1), "f1"),
    ]
    evals = ["NYC eval", "SF eval"]
    ops = ["NYC op", "SF op"]

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)

    for r, (col, label, xlim, panel_kind) in enumerate(metrics):
        for c, ev in enumerate(evals):
            ax = axes[r, c]

            if panel_kind == "sed":
                budget = SED_BUDGET[ev]
                ax.axvspan(budget, xlim[1], color=BUDGET_FACE, alpha=0.07, zorder=0)
                ax.axvline(budget, color="black", linestyle="--", linewidth=0.9, zorder=1)

            for op in ops:
                xs, ys = ecdf(cells[(op, ev)][col])
                if len(xs) == 0:
                    continue
                is_matched = MATCHED_EVAL[op] == ev
                ax.plot(
                    xs, ys,
                    color=OP_COLORS[op],
                    linewidth=1.7 if is_matched else 1.3,
                    linestyle="-" if is_matched else "--",
                    zorder=3,
                )
                med = float(np.median(xs))

                if panel_kind == "sed":
                    frac, n = fraction_above(cells[(op, ev)][col], SED_BUDGET[ev])
                    y_text = 0.32 if op == "NYC op" else 0.18
                    ax.text(
                        0.97, y_text,
                        f"{op}: {frac * 100:.0f}\\% above $D$",
                        transform=ax.transAxes,
                        ha="right", va="center", fontsize=8, color=OP_COLORS[op],
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    )

                if panel_kind == "f1" and is_matched:
                    ax.axvline(med, color=OP_COLORS[op], linewidth=0.9, alpha=0.6, zorder=1)
                    ax.text(
                        med, 0.04, f"{med:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=OP_COLORS[op],
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                    )

            if panel_kind == "sed":
                ax.text(
                    SED_BUDGET[ev], 1.02,
                    f"$D = {SED_BUDGET[ev]:g}$ m",
                    ha="center", va="bottom", fontsize=8, color="#444444",
                    transform=ax.get_xaxis_transform(),
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, zorder=0)
            if r == 0:
                ax.set_title(ev)
            if c == 0:
                ax.set_ylabel(f"ECDF\n{label}")
            if r == 2:
                ax.set_xlabel(label)

    legend_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.7, label="matched (per-city) op"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.3, label="cross-city op"),
        mlines.Line2D([], [], color=OP_COLORS["NYC op"], linewidth=2.5, label="NYC op"),
        mlines.Line2D([], [], color=OP_COLORS["SF op"], linewidth=2.5, label="SF op"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=0.9,
                      label="SED budget $D$ (per eval city)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=True, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("HYSOC-G cross-city sensitivity: distributional view", y=1.05)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out_path = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
