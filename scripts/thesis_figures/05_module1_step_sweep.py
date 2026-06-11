"""Module I STEP trade-off figure: Stop F1 vs segment count, NYC | SF.

One panel per city plotting Stop F1 against median segment count, with each
(D, T) configuration coloured by the dwell threshold T and the selected
operating point boxed in red.
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
SWEEP_OUTPUT_ROOT = os.path.join(project_root, "results", "sweeps", "module1_step_stss")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

DPI = 300
F1_FLOOR = 0.6
MIN_SEGMENTS = 4.0
T_ORDER = [10.0, 15.0, 30.0, 60.0]
T_LABELS = {10.0: "$T$ = 10 s", 15.0: "$T$ = 15 s", 30.0: "$T$ = 30 s", 60.0: "$T$ = 60 s"}


def build_points(rows: list[dict], winner_d: float, winner_t: float) -> list[dict]:
    points = []
    for r in rows:
        d = to_float(r["eps_m"])
        t = to_float(r["t_s"])
        f1 = to_float(r["f1_median"])
        stops = to_float(r["step_n_stops_median"])
        moves = to_float(r["step_n_moves_median"])
        feasible = f1 >= F1_FLOOR and stops >= 2 and moves >= 2
        points.append({
            "x": stops + moves, "y": f1, "color": t,
            "feasible": feasible,
            "winner": abs(d - winner_d) < 1e-9 and abs(t - winner_t) < 1e-9,
            "label": None,
        })
    return points


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    nyc = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "nyc"))
    sf = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "sf"))
    panels = [
        ("NYC", build_points(nyc, OPERATING_POINTS["nyc"].stop_max_eps_m, OPERATING_POINTS["nyc"].stop_min_duration_s)),
        ("SF", build_points(sf, OPERATING_POINTS["sf"].stop_max_eps_m, OPERATING_POINTS["sf"].stop_min_duration_s)),
    ]

    fig = render_tradeoff(
        panels,
        x_label="Segment count (stops + moves, median)",
        y_label="Stop $F_1$ (median)",
        color_mode="discrete",
        discrete_order=T_ORDER,
        discrete_labels=T_LABELS,
        thresholds=[
            {"axis": "y", "value": F1_FLOOR, "label": rf"$F_1$ floor = {F1_FLOOR:g}"},
            {"axis": "x", "value": MIN_SEGMENTS, "label": "min. segments"},
        ],
        annotate=False,
        legend_ncol=4,
    )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_out = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_out, exist_ok=True)
    out = os.path.join(run_out, f"{SCRIPT_NAME}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    logger.info("Saved figure: %s", out)


if __name__ == "__main__":
    main()
