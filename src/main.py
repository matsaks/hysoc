from __future__ import annotations

import argparse
from pathlib import Path

from core.compression import CompressionStrategy, HYSOCConfig
from hysoc.hysocG import HYSOCCompressor
from core.stream import TrajectoryStream


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HYSOC/Oracle orchestrator entrypoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV trajectory")
    parser.add_argument("--mode", choices=["hysoc_g", "hysoc_n"], default="hysoc_g")
    return parser


def run_hysoc(input_path: Path, mode: str) -> None:
    strategy = CompressionStrategy.GEOMETRIC if mode == "hysoc_g" else CompressionStrategy.NETWORK_SEMANTIC
    config = HYSOCConfig(move_compression_strategy=strategy)
    stream = TrajectoryStream(filepath=input_path, col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"})
    points = list(stream.stream())
    result = HYSOCCompressor(config=config).compress(points)
    print(f"[{mode}] Compressed {result.total_original_points} -> {result.total_compressed_points} points")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_hysoc(Path(args.input), args.mode)


if __name__ == "__main__":
    main()
