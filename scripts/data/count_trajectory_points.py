"""Count points per trajectory in a NYC dataset and report min/max/mean/median."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constants.dataset_paths import CALIBRATION_DIR, EVALUATION_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("calibration", "evaluation"),
        default="calibration",
        help="Which dataset to count (default: calibration).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override path to a directory of trajectory CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif args.dataset == "evaluation":
        data_dir = EVALUATION_DIR
    else:
        data_dir = CALIBRATION_DIR

    counts: dict[str, int] = {}

    for csv_file in data_dir.glob("*.csv"):
        with csv_file.open() as f:
            # subtract 1 for the header line
            counts[csv_file.name] = sum(1 for _ in f) - 1

    if not counts:
        print(f"No CSV files found in {data_dir}.")
        return

    min_name = min(counts, key=lambda k: counts[k])
    max_name = max(counts, key=lambda k: counts[k])

    print(f"Dataset: {data_dir}")
    print(f"Total trajectories: {len(counts)}")
    print(f"Total points:       {sum(counts.values())}")
    print()
    print(f"Fewest points:  {min_name}  ->  {counts[min_name]} points")
    print(f"Most points:    {max_name}  ->  {counts[max_name]} points")
    print()

    sorted_counts = sorted(counts.values())
    median_val = sorted_counts[len(sorted_counts) // 2]
    mean_val = sum(sorted_counts) / len(sorted_counts)
    print(f"Mean:   {mean_val:.1f} points")
    print(f"Median: {median_val} points")


if __name__ == "__main__":
    main()
