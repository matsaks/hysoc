"""
Split raw vehicle GPS data into one CSV per vehicle ID.

Input:  tab-separated file with columns:
        row_id, vehicle_id, date, time, time, time, xpos (lon), ypos (lat), speed

Output: one CSV per vehicle_id written to data/raw/<dataset_name>/
        columns: time, latitude, longitude, speed
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split vehicle GPS data by vehicle ID")
    parser.add_argument("input_file", help="Path to the raw tab-separated GPS file")
    parser.add_argument("output_dir", help="Directory to write per-vehicle CSV files")
    return parser.parse_args()


def parse_timestamp(date_str: str, time_str: str) -> str:
    dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def split_by_vehicle(input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Collect rows per vehicle in a single pass
    vehicle_rows: dict[str, list[list[str]]] = defaultdict(list)

    with open(input_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header

        for row in reader:
            if len(row) < 9:
                continue
            _, vehicle_id, date, time1, _, _, xpos, ypos, speed = row[:9]
            timestamp = parse_timestamp(date.strip(), time1.strip())
            vehicle_rows[vehicle_id.strip()].append(
                [timestamp, ypos.strip(), xpos.strip(), speed.strip()]
            )

    out_header = ["time", "latitude", "longitude", "speed"]
    written = 0
    for vehicle_id, rows in vehicle_rows.items():
        out_path = os.path.join(output_dir, f"{vehicle_id}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_header)
            writer.writerows(rows)
        written += 1

    print(f"Written {written} files to {output_dir}")
    for vid, rows in sorted(vehicle_rows.items(), key=lambda x: int(x[0])):
        print(f"  vehicle {vid}: {len(rows):,} points")


def main() -> None:
    args = parse_args()
    split_by_vehicle(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
