"""
Extract NYC or Tokyo subsets from the WorldTrace corpus.

Selects the 1,100 longest trajectories whose first GPS fix falls inside the
target region, then splits them into a 100-trajectory calibration set and a
1,000-trajectory evaluation set.

The script reads two zipped artefacts that ship with WorldTrace:
    Meta.zip        -- one JSON per trajectory with fields `geometry` (a list of
                       [lon, lat] pairs) and `Points` (the trajectory length).
    Trajectory.zip  -- one CSV per trajectory with the full per-point record
                       (time, latitude, longitude, altitude, osm_way_id, ...).

The WorldTrace archive is assumed to live in a Google Drive folder mounted at
`/content/drive/MyDrive/WorldTrace` (the default `--base-path`).  When running
outside Colab, point `--base-path` at a local copy of that folder.  All output
folders are created under `--output-dir`, which defaults to the same base path.

The original Colab extraction did not seed `random.shuffle`, so the
calibration/evaluation split of the existing on-disk datasets is not bit-exact
reproducible.  This script accepts `--seed` to make future re-runs
deterministic.

Usage:
    uv run python scripts/data/prepare_worldtrace.py --region nyc
    uv run python scripts/data/prepare_worldtrace.py --region tokyo --seed 0
    uv run python scripts/data/prepare_worldtrace.py --region nyc \\
        --base-path /path/to/WorldTrace --output-dir data/raw
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass

from shapely.geometry import Point, Polygon, box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_PATH = "/content/drive/MyDrive/WorldTrace"
META_ARCHIVE = "Meta.zip"
TRAJECTORY_ARCHIVE = "Trajectory.zip"

TARGET_COUNT = 1100
CALIBRATION_SIZE = 100
EVALUATION_SIZE = TARGET_COUNT - CALIBRATION_SIZE

SCAN_PROGRESS_INTERVAL = 20_000
EXTRACT_PROGRESS_INTERVAL = 100

# NYC polygon (GeoJSON CRS84: [lon, lat] ordering).
NYC_POLYGON = Polygon([
    [-73.975284120856401, 40.810159695562334],
    [-74.011508897209907, 40.755644485957674],
    [-74.023281949525014, 40.708636378828174],
    [-74.020112281594109, 40.696278373761828],
    [-74.005622371052326, 40.697994900465147],
    [-73.993396509033374, 40.707263380382784],
    [-73.974831311152101, 40.708636378828174],
    [-73.966680736472640, 40.726482782519781],
    [-73.966680736472640, 40.743295207600227],
    [-73.948115538591367, 40.762847172521674],
    [-73.940417773616190, 40.772106622733247],
    [-73.938606534798581, 40.781364782736716],
    [-73.931361579528158, 40.790278828453665],
    [-73.975284120856401, 40.810159695562334],
])

# Tokyo bounding box: 23 special wards plus surrounding greater Tokyo
# (Shinjuku, Shibuya, Chiyoda, Minato, Setagaya, Adachi, ...).
TOKYO_POLYGON = box(139.55, 35.52, 139.95, 35.85)

# San Francisco central districts: Pacific Heights / Marina south through
# the Mission / Castro and out to Inner Sunset / Twin Peaks. Hand-drawn to
# exclude the surrounding bay-area highway network so the trajectory mix
# stays urban rather than dominated by inter-city through-routes.
SF_CENTRE_POLYGON = Polygon([
    [-122.445278945349472, 37.803056405274049],
    [-122.444674288446748, 37.789964815443895],
    [-122.461725613105557, 37.787862300161038],
    [-122.464023309336099, 37.774959192036988],
    [-122.452655759563413, 37.774576843364244],
    [-122.451809239899504, 37.766164672309756],
    [-122.451567377138332, 37.756699836005865],
    [-122.449390612288596, 37.745608183941101],
    [-122.431976493487809, 37.744173797194378],
    [-122.419520561290454, 37.742165609047220],
    [-122.414320511926263, 37.740348629966149],
    [-122.409241393942850, 37.741687460981908],
    [-122.406943697712322, 37.745321308815818],
    [-122.407306491853873, 37.750102415730311],
    [-122.408878599801284, 37.756891056806523],
    [-122.410329776367917, 37.764157080964821],
    [-122.410692570509468, 37.769128159931441],
    [-122.409362325323229, 37.776488566954171],
    [-122.403436687675921, 37.780598605272090],
    [-122.396059873461979, 37.787288876519966],
    [-122.393157520328302, 37.791876141060840],
    [-122.388199333725694, 37.799616504201651],
    [-122.398720363834059, 37.812037613879340],
    [-122.431613699346258, 37.813470684403654],
    [-122.445278945349472, 37.803056405274049],
])


@dataclass(frozen=True)
class _Region:
    name: str
    polygon: Polygon
    calibration_dir: str
    evaluation_dir: str


REGIONS: dict[str, _Region] = {
    "nyc": _Region(
        name="NYC",
        polygon=NYC_POLYGON,
        calibration_dir="NYC_Calibration_100",
        evaluation_dir="NYC_Evaluation_1000",
    ),
    "tokyo": _Region(
        name="Tokyo",
        polygon=TOKYO_POLYGON,
        calibration_dir="Tokyo_Calibration_100",
        evaluation_dir="Tokyo_Evaluation_1000",
    ),
    "sf_centre": _Region(
        name="SanFranCentre",
        polygon=SF_CENTRE_POLYGON,
        calibration_dir="SanFranCentre_Calibration_100",
        evaluation_dir="SanFranCentre_Evaluation_1000",
    ),
}


@dataclass(frozen=True)
class _TrajectoryMeta:
    id: str
    points: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_meta(meta_zip_path: str, polygon: Polygon) -> list[_TrajectoryMeta]:
    """Return trajectories whose first GPS fix lies inside `polygon`."""
    matches: list[_TrajectoryMeta] = []

    with zipfile.ZipFile(meta_zip_path, "r") as z_meta:
        meta_files = [f for f in z_meta.namelist() if f.endswith(".json")]
        total = len(meta_files)

        for i, meta_file in enumerate(meta_files):
            if i > 0 and i % SCAN_PROGRESS_INTERVAL == 0:
                print(f"  scanned {i}/{total} meta files (kept {len(matches)})")

            with z_meta.open(meta_file) as f:
                try:
                    data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

            geom = data.get("geometry") or []
            if not geom:
                continue

            lon, lat = geom[0]
            if not polygon.contains(Point(lon, lat)):
                continue

            file_id = os.path.basename(meta_file).removesuffix(".json")
            matches.append(_TrajectoryMeta(id=file_id, points=int(data.get("Points", 0))))

    return matches


def _extract_set(
    trajectories: list[_TrajectoryMeta],
    traj_zip_path: str,
    destination_dir: str,
    label: str,
) -> int:
    """Copy each trajectory CSV from `traj_zip_path` to `destination_dir`."""
    os.makedirs(destination_dir, exist_ok=True)

    with zipfile.ZipFile(traj_zip_path, "r") as z_traj:
        path_map = {
            os.path.basename(p): p
            for p in z_traj.namelist()
            if p.endswith(".csv")
        }

        written = 0
        missing: list[str] = []

        for count, traj in enumerate(trajectories, start=1):
            csv_filename = f"{traj.id}.csv"
            internal_path = path_map.get(csv_filename)
            if internal_path is None:
                missing.append(csv_filename)
                continue

            dest_path = os.path.join(destination_dir, csv_filename)
            with z_traj.open(internal_path) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1

            if count % EXTRACT_PROGRESS_INTERVAL == 0:
                print(f"  {label}: extracted {count}/{len(trajectories)}")

    if missing:
        print(f"  WARNING {label}: {len(missing)} CSV(s) not found in {traj_zip_path}")
        for name in missing[:5]:
            print(f"    missing: {name}")
        if len(missing) > 5:
            print(f"    (+{len(missing) - 5} more)")

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 1,100 longest WorldTrace trajectories in a target region "
            "and split them into a 100/1,000 calibration/evaluation pair."
        )
    )
    parser.add_argument(
        "--region",
        choices=sorted(REGIONS.keys()),
        required=True,
        help="Region to extract (nyc or tokyo).",
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=(
            "Folder containing Meta.zip and Trajectory.zip. "
            f"Defaults to the Google Drive mount path: {DEFAULT_BASE_PATH}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write the calibration and evaluation subfolders. "
            "Defaults to --base-path."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random shuffle (default: 0).",
    )
    args = parser.parse_args()

    region = REGIONS[args.region]
    meta_path = os.path.join(args.base_path, META_ARCHIVE)
    traj_path = os.path.join(args.base_path, TRAJECTORY_ARCHIVE)
    output_dir = args.output_dir or args.base_path

    for required in (meta_path, traj_path):
        if not os.path.isfile(required):
            print(f"Required archive not found: {required}")
            sys.exit(1)

    calibration_dir = os.path.join(output_dir, region.calibration_dir)
    evaluation_dir = os.path.join(output_dir, region.evaluation_dir)

    print(f"Region        : {region.name}")
    print(f"Base path     : {args.base_path}")
    print(f"Output dir    : {output_dir}")
    print(f"Seed          : {args.seed}")
    print(f"Calibration   : {calibration_dir}")
    print(f"Evaluation    : {evaluation_dir}")
    print()

    print("Scanning Meta.zip for in-region trajectories...")
    matches = _scan_meta(meta_path, region.polygon)
    print(f"Found {len(matches)} {region.name} trajectories.")

    if len(matches) < TARGET_COUNT:
        print(
            f"Not enough trajectories: need {TARGET_COUNT}, found {len(matches)}."
        )
        sys.exit(1)

    matches.sort(key=lambda t: t.points, reverse=True)
    top = matches[:TARGET_COUNT]

    rng = random.Random(args.seed)
    rng.shuffle(top)

    calibration_list = top[:CALIBRATION_SIZE]
    evaluation_list = top[CALIBRATION_SIZE:]

    print(f"\nExtracting {len(calibration_list)} trajectories to calibration set...")
    n_cal = _extract_set(calibration_list, traj_path, calibration_dir, "calibration")

    print(f"\nExtracting {len(evaluation_list)} trajectories to evaluation set...")
    n_eval = _extract_set(evaluation_list, traj_path, evaluation_dir, "evaluation")

    print()
    print(f"Done. Calibration: {n_cal} files at {calibration_dir}")
    print(f"      Evaluation : {n_eval} files at {evaluation_dir}")


if __name__ == "__main__":
    main()
