"""
HYSOC-N (network-semantic) pipeline wrapper.
"""

from core.compression import CompressionStrategy, HYSOCConfig
from .hysocG import HYSOCGCompressor


class HYSOCNCompressor(HYSOCGCompressor):
    """HYSOC-N pipeline with network-semantic defaults."""

    def __init__(self, config: HYSOCConfig | None = None):
        cfg = config if config is not None else HYSOCConfig(
            move_compression_strategy=CompressionStrategy.NETWORK_SEMANTIC
        )
        if cfg.move_compression_strategy != CompressionStrategy.NETWORK_SEMANTIC:
            cfg.move_compression_strategy = CompressionStrategy.NETWORK_SEMANTIC
        super().__init__(config=cfg)
