"""
Flat registry of shared trajectory algorithms used across HYSOC and oracle pipelines.

One file per algorithm:
    dp             - Ramer-Douglas-Peucker line simplification
    squish         - SQUISH SED-based priority-queue geometric compressor
    squish_dp      - Hybrid SQUISH + DP refinement compressor
    trace          - TRACE network-semantic k-mer referential compressor
    step           - STEP streaming stay-point segmenter
    stop_compressor- Stop centroid/duration compressor
    hmm            - Online HMM map matcher (Viterbi sliding window)
    map_matched_stream - Stream wrapper that injects map-matched road_ids
    stss_sklearn   - STSS (OPTICS, sklearn) offline density-based segmenter
    stss_manual    - STSS (manual DBSCAN-like) offline density-based segmenter
    stc            - STC offline network-semantic move compressor
"""

from .dp import DouglasPeuckerCompressor
from .hmm import OnlineMapMatcher
from .map_matched_stream import MapMatchedStreamWrapper
from .squish import SquishCompressor
from .squish_dp import HybridSquishDPCompressor, HybridSquishDPConfig
from .stc import STCOracle
from .step import STEPSegmenter
from .stop_compressor import CompressedStop, StopCompressor
from .stss_manual import STSSOracleManual
from .stss_sklearn import STSSOracleSklearn
from .trace import Reference, TraceCompressor

__all__ = [
    "CompressedStop",
    "DouglasPeuckerCompressor",
    "HybridSquishDPCompressor",
    "HybridSquishDPConfig",
    "MapMatchedStreamWrapper",
    "OnlineMapMatcher",
    "Reference",
    "STCOracle",
    "STEPSegmenter",
    "STSSOracleManual",
    "STSSOracleSklearn",
    "SquishCompressor",
    "StopCompressor",
    "TraceCompressor",
]
