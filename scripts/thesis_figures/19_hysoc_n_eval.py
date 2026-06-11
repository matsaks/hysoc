"""HYSOC-N end-to-end evaluation figure: 2x4 per-trajectory distributions.

Boxplots comparing Plain TRACE, Baseline-N, and HYSOC-N across CR, road-segment
Jaccard, Stop F1, and latency (rows: NYC, SF). The Stop F1 panel draws HYSOC-N
only, since Plain TRACE has no stops and Baseline-N is trivially 1.0.
"""

# ruff: noqa: E402

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

from _latex_style import apply_latex_style

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
EVAL_OUTPUT_ROOT = os.path.join(
    project_root, "results", "experiments", "hysoc_n_eval"
)
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")

NYC_RUN_DIR: str | None = None
SF_RUN_DIR: str | None = None

USE_LATEX = True
DPI = 300
FIGSIZE = (6.5, 8.0)

PIPELINES = ["plain_trace", "baseline_n", "hysoc_n"]
PIPELINE_LABELS = {
    "plain_trace": "Plain TRACE",
    "baseline_n": "Baseline-N",
    "hysoc_n": "HYSOC-N",
}
PIPELINE_COLORS = {
    "plain_trace": "#bdbdbd",
    "baseline_n": "#64b5f6",
    "hysoc_n": "#1565c0",
}

CITIES = [
    ("NYC", "nyc"),
    ("SF", "sf_centre"),
]

COLUMNS = [
    ("compression_ratio", "Compression Ratio", False, False),
    ("road_jaccard_vs_truth", "Road-segment Jaccard", False, False),
    ("stop_f1", r"Stop $F_1$", False, True),
    ("latency_median_us_per_point", r"Latency ($\mu$s / point)", True, False),
]


def _resolve_run_dir(
    override: str | None, city_slug: str, logger: logging.Logger
) -> str:
    if not os.path.isdir(EVAL_OUTPUT_ROOT):
        logger.error(f"Eval output root missing: {EVAL_OUTPUT_ROOT}")
        sys.exit(1)
    if override is not None:
        path = os.path.join(EVAL_OUTPUT_ROOT, override)
        if not os.path.isdir(path):
            logger.error(f"Override run dir not found: {path}")
            sys.exit(1)
        return path
    suffix = f"_op-{city_slug}"
    candidates = [
        d for d in os.listdir(EVAL_OUTPUT_ROOT)
        if d.endswith(suffix)
        and os.path.isdir(os.path.join(EVAL_OUTPUT_ROOT, d))
    ]
    if not candidates:
        logger.error(f"No run dirs ending in {suffix} under {EVAL_OUTPUT_ROOT}")
        sys.exit(1)
    return os.path.join(EVAL_OUTPUT_ROOT, max(candidates))


def _load_per_trajectory(
    run_dir: str, city_slug: str, logger: logging.Logger
) -> list[dict]:
    csv_path = os.path.join(run_dir, f"{city_slug}_per_trajectory.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Missing per-trajectory CSV: {csv_path}")
        sys.exit(1)
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Loaded {len(rows)} rows from {csv_path}")
    return rows


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _split_by_pipeline(rows: list[dict], column: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {p: [] for p in PIPELINES}
    for r in rows:
        p = r.get("pipeline")
        if p not in out:
            continue
        v = _to_float(r.get(column, ""))
        if not math.isnan(v):
            out[p].append(v)
    return out


_BOX_KWARGS = {
    "widths": 0.55,
    "showmeans": True,
    "patch_artist": True,
    "meanprops": {
        "marker": "^",
        "markerfacecolor": "#2ca02c",
        "markeredgecolor": "#1b5e1b",
        "markersize": 5,
    },
    "medianprops": {"color": "#ff7f0e", "linewidth": 1.5},
    "flierprops": {
        "marker": "o",
        "markersize": 2.0,
        "markerfacecolor": "none",
        "markeredgecolor": "#444444",
        "alpha": 0.4,
    },
}


def _draw_box_panel(
    ax,
    data_by_pipeline: dict[str, list[float]],
    *,
    log_scale: bool,
) -> None:
    positions = list(range(1, len(PIPELINES) + 1))
    plot_data, plot_positions, plot_colors = [], [], []
    for pos, pipe in zip(positions, PIPELINES):
        if data_by_pipeline[pipe]:
            plot_data.append(data_by_pipeline[pipe])
            plot_positions.append(pos)
            plot_colors.append(PIPELINE_COLORS[pipe])

    if plot_data:
        bp = ax.boxplot(plot_data, positions=plot_positions, **_BOX_KWARGS)
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels([PIPELINE_LABELS[p] for p in PIPELINES])
    ax.set_xlim(0.4, len(PIPELINES) + 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    if log_scale:
        ax.set_yscale("log")


def _draw_stop_f1_panel(ax, hysoc_values: list[float]) -> None:
    if hysoc_values:
        bp = ax.boxplot(hysoc_values, positions=[1], **_BOX_KWARGS)
        bp["boxes"][0].set_facecolor(PIPELINE_COLORS["hysoc_n"])
        bp["boxes"][0].set_alpha(0.75)
    ax.axhline(1.0, linestyle="--", color="#888888", linewidth=0.9, alpha=0.8)
    ax.set_xticks([1])
    ax.set_xticklabels([PIPELINE_LABELS["hysoc_n"]])
    ax.set_xlim(0.3, 1.7)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(SCRIPT_NAME)

    apply_latex_style(use_latex=USE_LATEX)
    import matplotlib.pyplot as plt

    runs = {
        "nyc": _resolve_run_dir(NYC_RUN_DIR, "nyc", logger),
        "sf_centre": _resolve_run_dir(SF_RUN_DIR, "sf_centre", logger),
    }
    logger.info(f"NYC run: {runs['nyc']}")
    logger.info(f"SF  run: {runs['sf_centre']}")

    fig, axes = plt.subplots(
        len(COLUMNS), len(CITIES), figsize=FIGSIZE, constrained_layout=True
    )

    traj_rows_by_city: dict[str, list[dict]] = {}
    n_traj_per_city: dict[str, int] = {}
    for city_label, city_slug in CITIES:
        traj_rows = _load_per_trajectory(runs[city_slug], city_slug, logger)
        traj_rows_by_city[city_slug] = traj_rows
        n_traj_per_city[city_label] = sum(
            1 for r in traj_rows if r.get("pipeline") == "hysoc_n"
        )

    for row_idx, (key, title, log_scale, is_stop_f1) in enumerate(COLUMNS):
        for col_idx, (city_label, city_slug) in enumerate(CITIES):
            ax = axes[row_idx][col_idx]
            data = _split_by_pipeline(traj_rows_by_city[city_slug], key)
            if is_stop_f1:
                _draw_stop_f1_panel(ax, data["hysoc_n"])
            else:
                _draw_box_panel(ax, data, log_scale=log_scale)
            if key == "compression_ratio":
                ax.set_ylim(top=250)
            if row_idx == 0:
                ax.set_title(city_label, fontweight="bold")
        axes[row_idx][0].set_ylabel(title)

    n_str = " / ".join(
        f"{n_traj_per_city[label]} {label}" for label, _ in CITIES
    )
    fig.suptitle(
        f"HYSOC-N evaluation -- per-trajectory distributions ({n_str})"
    )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
