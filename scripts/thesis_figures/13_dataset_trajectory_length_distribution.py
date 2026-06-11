"""Side-by-side histograms of trajectory length for the NYC and SF Centre datasets.

Counts GPS points per trajectory across the full 1,100-trajectory set per region
and renders two shared-y histograms with the per-region median overlaid. Counts
are cached; set FORCE_RECOMPUTE=True to refresh.
"""

# ruff: noqa: E402

import csv
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
RAW_ROOT = os.path.join(project_root, "data", "raw")
OUTPUT_ROOT = os.path.join(project_root, "results", "figures")
CACHE_PATH = os.path.join(OUTPUT_ROOT, f"{SCRIPT_NAME}_cache.csv")
FORCE_RECOMPUTE = False

DATASETS: list[tuple[str, str, list[str]]] = [
    ("NYC", "NYC", ["NYC_Calibration_100", "NYC_Evaluation_1000"]),
    (
        "SF Centre",
        "SF_Centre",
        ["SanFranCentre_Calibration_100", "SanFranCentre_Evaluation_1000"],
    ),
]

BAR_COLOR = "#4a4a4a"
MEDIAN_COLOR = "#b22222"
N_BINS = 60
HIST_PERCENTILE_CLIP = 99.0
FIG_SIZE = (11, 4.2)
DPI = 300


def count_points(path: str) -> int:
    """Return the number of data rows (points) in a trajectory CSV."""
    with open(path, "r", newline="") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def scan_datasets(logger: logging.Logger) -> list[dict]:
    records: list[dict] = []
    for label, key, subdirs in DATASETS:
        for subdir in subdirs:
            data_dir = os.path.join(RAW_ROOT, subdir)
            if not os.path.isdir(data_dir):
                logger.warning(f"Missing dataset directory: {data_dir}")
                continue
            csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
            logger.info(f"{label} / {subdir}: {len(csv_files)} files")
            for fname in csv_files:
                obj_id = fname.replace(".csv", "")
                n_pts = count_points(os.path.join(data_dir, fname))
                records.append(
                    {"dataset": key, "label": label, "obj_id": obj_id, "n_points": n_pts}
                )
    return records


def write_cache(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "label", "obj_id", "n_points"])
        writer.writeheader()
        writer.writerows(records)


def read_cache(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def plot_distribution(records: list[dict], logger: logging.Logger) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_latex_style(use_latex=True)

    by_key: dict[str, list[int]] = {}
    label_for: dict[str, str] = {}
    for r in records:
        n = int(r["n_points"])
        by_key.setdefault(r["dataset"], []).append(n)
        label_for[r["dataset"]] = r["label"]

    ordered_keys = [key for _, key, _ in DATASETS if key in by_key]
    all_counts = np.concatenate([np.array(by_key[k]) for k in ordered_keys])
    upper = float(np.percentile(all_counts, HIST_PERCENTILE_CLIP))
    bins = np.linspace(0, upper, N_BINS + 1)

    fig, axes = plt.subplots(
        1, len(ordered_keys), figsize=FIG_SIZE, sharey=True, sharex=True
    )
    if len(ordered_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, ordered_keys):
        counts = np.array(by_key[key])
        clipped = counts[counts <= upper]
        median = float(np.median(counts))
        ax.hist(clipped, bins=bins, color=BAR_COLOR, edgecolor="none")
        ax.axvline(
            median,
            color=MEDIAN_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"median = {median:.0f}",
        )
        ax.set_title(label_for[key])
        ax.set_xlabel("Trajectory length (points)")
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        logger.info(
            f"{label_for[key]}: n={len(counts)}, median={median:.0f}, "
            f"mean={counts.mean():.1f}, max={counts.max()}"
        )

    axes[0].set_ylabel("Trajectory count")
    fig.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_ROOT, SCRIPT_NAME, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{SCRIPT_NAME}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("dataset_trajectory_length_distribution")

    if not FORCE_RECOMPUTE and os.path.exists(CACHE_PATH):
        logger.info(f"Loading cached counts from {CACHE_PATH}")
        records = read_cache(CACHE_PATH)
        logger.info(f"Loaded {len(records)} cached records")
    else:
        records = scan_datasets(logger)
        write_cache(records, CACHE_PATH)
        logger.info(f"Cached {len(records)} records to {CACHE_PATH}")

    if not records:
        logger.error("No trajectories found.")
        return

    plot_distribution(records, logger)


if __name__ == "__main__":
    main()
