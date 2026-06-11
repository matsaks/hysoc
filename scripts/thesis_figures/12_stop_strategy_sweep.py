"""Module II stop-compressor trade-off figure: latency vs median mean SED, NYC | SF.

One panel per city plotting per-point latency (log axis) against median mean SED
for the four single-keypoint strategies, with the 15 m SED budget marked and the
selected centroid boxed in red.
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

from constants.stop_compression_defaults import STOP_COMPRESSION_DEFAULT_STRATEGY

from _tradeoff_panels import latest_run, load_aggregated, render_tradeoff, to_float

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SWEEP_OUTPUT_ROOT = os.path.join(project_root, "results", "sweeps", "module2_stop_compression")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

DPI = 300
BUDGET_M = 15.0
STRATEGY_ORDER = ["centroid", "snap_to_nearest", "medoid", "first_point"]
STRATEGY_LABELS = {
    "centroid": "centroid",
    "snap_to_nearest": "snap-to-nearest",
    "medoid": "medoid",
    "first_point": "first-point",
}


def build_points(rows: list[dict], winner_strategy: str) -> list[dict]:
    by_strategy = {r["strategy"]: r for r in rows}
    points = []
    for strategy in STRATEGY_ORDER:
        r = by_strategy[strategy]
        sed = to_float(r["avg_sed_m_median"])
        lat = to_float(r["latency_us_per_point_median"])
        points.append({
            "x": sed, "y": lat, "color": strategy,
            "feasible": sed <= BUDGET_M,
            "winner": strategy == winner_strategy,
            "label": None,
        })
    return points


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(SCRIPT_NAME)

    selected = STOP_COMPRESSION_DEFAULT_STRATEGY.value
    nyc = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "nyc"))
    sf = load_aggregated(latest_run(SWEEP_OUTPUT_ROOT, "sf"))
    panels = [
        ("NYC", build_points(nyc, selected)),
        ("SF", build_points(sf, selected)),
    ]

    fig = render_tradeoff(
        panels,
        x_label="Mean SED across stops (m, median)",
        y_label=r"Latency ($\mu$s/point, median)",
        y_log=True,
        color_mode="discrete",
        discrete_order=STRATEGY_ORDER,
        discrete_labels=STRATEGY_LABELS,
        thresholds=[{"axis": "x", "value": BUDGET_M,
                     "label": rf"SED budget = {BUDGET_M:g} m", "infeasible": "greater"}],
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
