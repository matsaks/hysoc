"""Run the three constant-selection sweeps in sequence under one shared timestamp."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sweeps import module1_step_stss, module2_stop_compression, module3_hysoc_g_move
from sweeps._common import DATASET_REGISTRY, DEFAULT_DATASET, RUN_ID_FORMAT


SWEEPS = [
    ("module1_step_stss", module1_step_stss),
    ("module2_stop_compression", module2_stop_compression),
    ("module3_hysoc_g_move", module3_hysoc_g_move),
]


def _invoke(
    module,
    max_trajectories: int | None,
    out_dir: str | None,
    run_id: str,
    dataset: str,
) -> None:
    argv = [module.__name__, "--run-id", run_id, "--dataset", dataset]
    if max_trajectories is not None:
        argv += ["--max-trajectories", str(max_trajectories)]
    if out_dir is not None:
        argv += ["--out-dir", out_dir]
    saved = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY),
        default=DEFAULT_DATASET,
        help="Calibration dataset to sweep on (default: nyc).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit MMDD_HHMM-style run id shared across sweeps (default: fresh timestamp).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap number of trajectories per sweep (smoke tests).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output base directory (defaults to results/sweeps/).",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _ in SWEEPS],
        action="append",
        default=None,
        help="Run only the named sweep(s); may be passed multiple times.",
    )
    args = parser.parse_args()

    selected = SWEEPS if not args.only else [(n, m) for n, m in SWEEPS if n in args.only]
    run_id = args.run_id if args.run_id is not None else datetime.now().strftime(RUN_ID_FORMAT)
    print(
        f"[run_all] Running {len(selected)} sweep(s): {[n for n, _ in selected]} "
        f"on dataset={args.dataset!r} under run_id={run_id}"
    )

    overall_start = time.perf_counter()
    for name, module in selected:
        print(f"\n========== {name} ==========")
        sweep_start = time.perf_counter()
        _invoke(module, args.max_trajectories, args.out_dir, run_id, args.dataset)
        print(f"[run_all] {name} took {time.perf_counter() - sweep_start:.1f}s")

    print(f"\n[run_all] All done in {time.perf_counter() - overall_start:.1f}s.")


if __name__ == "__main__":
    main()
