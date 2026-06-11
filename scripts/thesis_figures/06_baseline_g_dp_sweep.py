"""Baseline-G DP epsilon trade-off figure: CR vs mean SED, NYC | SF.

One panel per city showing the compression-versus-fidelity frontier across DP
tolerances, with the 15 m mean-SED budget marked and the selected operating
point boxed in red.
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

from constants.operating_points import OPERATING_POINTS

from _tradeoff_panels import latest_run, load_aggregated, render_tradeoff, to_float

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SWEEP_OUTPUT_ROOT = os.path.join(project_root, "results", "sweeps", "oracle_g_dp_epsilon")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

DPI = 300
BUDGET_M = 15.0
ANNOT_EPS = {1, 5, 10, 50}


def build_points(rows: list[dict], winner_eps: float) -> list[dict]:
    points = []
    for r in rows:
        eps = to_float(r["dp_epsilon_m"])
        sed = to_float(r["move_mean_sed_m_median"])
        cr = to_float(r["move_cr_median"])
        points.append({
            "x": sed, "y": cr, "color": eps,
            "feasible": sed <= BUDGET_M,
            "winner": abs(eps - winner_eps) < 1e-9,
            "label": rf"$\varepsilon{{=}}{eps:g}$" if eps in ANNOT_EPS else None,
        })
    return points


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    nyc = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "nyc"))
    sf = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "sf"))
    panels = [
        ("NYC", build_points(nyc, OPERATING_POINTS["nyc"].dp_epsilon_m)),
        ("SF", build_points(sf, OPERATING_POINTS["sf"].dp_epsilon_m)),
    ]

    fig = render_tradeoff(
        panels,
        x_label="Mean SED (m, median)",
        y_label="Move CR (median)",
        color_mode="continuous",
        color_label=r"DP tolerance $\varepsilon$ (m)",
        color_log=True, vmin=1, vmax=50,
        thresholds=[{"axis": "x", "value": BUDGET_M,
                     "label": rf"SED budget = {BUDGET_M:g} m", "infeasible": "greater"}],
        annotate=True,
    )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    logger.info("Saved figure: %s", out)


if __name__ == "__main__":
    main()
