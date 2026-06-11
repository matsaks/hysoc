"""HYSOC command-line entry point for one-shot compression and benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.stream import TrajectoryStream
from hysoc import HYSOCGCompressor, HYSOCNCompressor

from experiments.hysoc_g_eval import (
    build_arg_parser as build_hysoc_g_eval_parser,
    run as run_hysoc_g_eval,
)
from experiments.hysoc_n_eval import (
    build_arg_parser as build_hysoc_n_eval_parser,
    run as run_hysoc_n_eval,
)


def _run_compress(args: argparse.Namespace) -> None:
    stream = TrajectoryStream(
        filepath=Path(args.input),
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
    )
    points = list(stream.stream())
    compressor = HYSOCGCompressor() if args.mode == "hysoc_g" else HYSOCNCompressor()
    result = compressor.compress(points)
    print(f"[{args.mode}] Compressed {len(result.original_points)} -> {len(result.keypoints)} points")


def _run_experiment(args: argparse.Namespace) -> None:
    if args.which == "hysoc_g":
        run_hysoc_g_eval(args)
    elif args.which == "hysoc_n":
        run_hysoc_n_eval(args)
    else:
        raise ValueError(f"Unknown experiment: {args.which!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HYSOC orchestrator and experiment entry point")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    compress = subparsers.add_parser(
        "compress",
        help="Run a single trajectory through HYSOC-G or HYSOC-N and print the result.",
    )
    compress.add_argument("--input", type=str, required=True, help="Path to input CSV trajectory")
    compress.add_argument("--mode", choices=["hysoc_g", "hysoc_n"], default="hysoc_g")

    experiment = subparsers.add_parser(
        "experiment",
        help="Run a thesis-final benchmark. Supports: hysoc_g, hysoc_n.",
    )
    exp_sub = experiment.add_subparsers(dest="which", required=True)
    hysoc_g_eval_parser = exp_sub.add_parser(
        "hysoc_g",
        help="HYSOC-G vs Baseline-G vs Plain DP on NYC + SanFranCentre evaluation sets.",
    )
    build_hysoc_g_eval_parser(hysoc_g_eval_parser)
    hysoc_n_eval_parser = exp_sub.add_parser(
        "hysoc_n",
        help="HYSOC-N vs Baseline-N vs Plain TRACE on NYC + SanFranCentre evaluation sets.",
    )
    build_hysoc_n_eval_parser(hysoc_n_eval_parser)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cmd == "compress":
        _run_compress(args)
    elif args.cmd == "experiment":
        _run_experiment(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()
