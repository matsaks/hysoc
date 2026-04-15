from .hysocG import HYSOCGCompressor
from .hysocN import HYSOCNCompressor

# Backward-friendly alias for existing scripts that still call HYSOCCompressor
HYSOCCompressor = HYSOCGCompressor

__all__ = ["HYSOCGCompressor", "HYSOCNCompressor", "HYSOCCompressor"]
