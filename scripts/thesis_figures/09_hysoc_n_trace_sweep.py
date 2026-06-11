"""HYSOC-N TRACE trade-off figure: CR vs retention, NYC | SF.

One panel per city plotting move CR against the move-segment retention ratio,
coloured by the speed threshold gamma, with the 0.10 retention floor marked and
the selected (gamma = 5 m/s, k = 5) boxed in red.
"""

# ruff: noqa: E402

import logging
import os
import re
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from constants.operating_points import OPERATING_POINTS
from sweeps.module4_hysoc_n_trace import RETENTION_FLOOR

from _tradeoff_panels import latest_run, load_aggregated, render_tradeoff, to_float

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SWEEP_OUTPUT_ROOT = os.path.join(project_root, "results", "sweeps", "module4_hysoc_n_trace")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

# Canonical timestamp folders only, so ablation dirs are excluded from selection.
RUN_NAME_RE = re.compile(r"^(?:sf_)?\d{4}_\d{4}$")

DPI = 300


def build_points(rows: list[dict], winner_gamma: float, winner_k: int) -> list[dict]:
    points = []
    for r in rows:
        gamma = to_float(r["gamma_mps"])
        k = int(to_float(r["k"]))
        ret = to_float(r["move_retention_ratio_median"])
        cr = to_float(r["move_cr_median"])
        points.append({
            "x": ret, "y": cr, "color": gamma,
            "feasible": ret >= RETENTION_FLOOR,
            "winner": abs(gamma - winner_gamma) < 1e-9 and k == winner_k,
            "label": None,
        })
    return points


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    nyc = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "nyc", RUN_NAME_RE))
    sf = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "sf", RUN_NAME_RE))
    panels = [
        ("NYC", build_points(nyc, OPERATING_POINTS["nyc"].trace_gamma, int(OPERATING_POINTS["nyc"].trace_k))),
        ("SF", build_points(sf, OPERATING_POINTS["sf"].trace_gamma, int(OPERATING_POINTS["sf"].trace_k))),
    ]

    fig = render_tradeoff(
        panels,
        x_label="Retention ratio (median)",
        y_label="Move CR (median)",
        color_mode="continuous",
        color_label=r"Speed threshold $\gamma$ (m/s)",
        color_log=True, vmin=2, vmax=20,
        thresholds=[{"axis": "x", "value": RETENTION_FLOOR,
                     "label": rf"retention floor = {RETENTION_FLOOR:g}", "infeasible": "less"}],
        annotate=False,
    )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    logger.info("Saved figure: %s", out)


if __name__ == "__main__":
    main()
