"""Registry of shared trajectory algorithms used across HYSOC and oracle pipelines."""

from .dp import DouglasPeuckerCompressor
from .squish import SquishCompressor
from .squish_dp import HybridSquishDPCompressor
from .stc import STCOracle
from .step import STEPSegmenter
from .stop_compressor import CompressedStop, StopCompressor
from .stss import STSSOracle
from .trace import Reference, TraceCompressor

__all__ = [
    "CompressedStop",
    "DouglasPeuckerCompressor",
    "HybridSquishDPCompressor",
    "Reference",
    "STCOracle",
    "STEPSegmenter",
    "STSSOracle",
    "SquishCompressor",
    "StopCompressor",
    "TraceCompressor",
]
