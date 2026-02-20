from typing import List
from hysoc.core.point import Point

def calculate_compression_ratio(original: List[Point], compressed: List[Point]) -> float:
    """
    Calculates the compression ratio.
    Ratio = Original Count / Compressed Count.
    
    Args:
        original: List of original points.
        compressed: List of compressed points.
        
    Returns:
        Compression ratio (e.g., 10.0 for 10:1 compression). Returns 1.0 if compressed is empty.
    """
    if not compressed:
        return 1.0
    return len(original) / len(compressed)
