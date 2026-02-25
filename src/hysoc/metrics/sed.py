from typing import List, Dict
import math
from hysoc.core.point import Point

def calculate_sed_error(p_original: Point, p_start: Point, p_end: Point) -> float:
    """
    Calculates the Synchronized Euclidean Distance (SED) error for a single point
    relative to a segment defined by p_start and p_end.
    
    Args:
        p_original: The point to calculate error for.
        p_start: The start point of the compressed segment.
        p_end: The end point of the compressed segment.
        
    Returns:
        Euclidean distance between p_original and its temporal projection on the segment.
    """
    t_orig = p_original.timestamp.timestamp()
    t_start = p_start.timestamp.timestamp()
    t_end = p_end.timestamp.timestamp()

    if t_start == t_end:
        # If segment is a single point (or zero duration), distance is just geometric distance to start
        d_lat = p_original.lat - p_start.lat
        d_lon = p_original.lon - p_start.lon
        avg_lat = math.radians((p_original.lat + p_start.lat) / 2.0)
        d_lat_m = d_lat * 111320.0
        d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
        return math.sqrt(d_lat_m*d_lat_m + d_lon_m*d_lon_m)

    # Calculate temporal ratio
    ratio = (t_orig - t_start) / (t_end - t_start)
    
    # Linearly interpolate lat/lon
    pred_lat = p_start.lat + (p_end.lat - p_start.lat) * ratio
    pred_lon = p_start.lon + (p_end.lon - p_start.lon) * ratio
    
    # Calculate Euclidean distance in meters
    d_lat = p_original.lat - pred_lat
    d_lon = p_original.lon - pred_lon
    
    avg_lat = math.radians((p_start.lat + p_end.lat) / 2.0)
    d_lat_m = d_lat * 111320.0
    d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
    
    return math.sqrt(d_lat_m*d_lat_m + d_lon_m*d_lon_m)

def calculate_sed_stats(original: List[Point], compressed: List[Point]) -> Dict[str, float | List[float]]:
    """
    Calculates statistics related to Synchronized Euclidean Distance (SED) error.
    
    Metrics:
    - average_sed: Mean SED error over all original points.
    - max_sed: Maximum SED error encountered.
    - rmse: Root Mean Square Error.
    
    Args:
        original: List of original points.
        compressed: List of compressed points.
        
    Returns:
        Dictionary containing 'average_sed', 'max_sed', 'rmse', and 'sed_errors'.
    """
    if not original or not compressed:
        return {'average_sed': 0.0, 'max_sed': 0.0, 'rmse': 0.0, 'sed_errors': []}
        
    sed_errors = []
    
    # For each point in original, find the corresponding segment in compressed
    # We assume compressed is a subset of original, sorted by time
    
    comp_idx = 0
    # compressed[comp_idx] is the start of the current segment
    # compressed[comp_idx+1] is the end of the current segment
    
    for p in original:
        # Find the segment [comp_idx, comp_idx+1] that covers p.timestamp
        # Advance comp_idx if necessary
        while comp_idx < len(compressed) - 1 and p.timestamp > compressed[comp_idx+1].timestamp:
            comp_idx += 1
            
        if comp_idx >= len(compressed) - 1:
            # We are past the last segment, or at the very last point
            # If p is exactly the last point, error is 0.
            # If p is somehow beyond the last point (shouldn't happen if subset), calculate dist to last point
            p_ref = compressed[-1]
            d_lat = p.lat - p_ref.lat
            d_lon = p.lon - p_ref.lon
            avg_lat = math.radians((p.lat + p_ref.lat) / 2.0)
            d_lat_m = d_lat * 111320.0
            d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
            dist = math.sqrt(d_lat_m*d_lat_m + d_lon_m*d_lon_m)
            sed_errors.append(dist)
            continue
            
        p_start = compressed[comp_idx]
        p_end = compressed[comp_idx+1]
        
        # Check if p is before p_start (shouldn't happen if sorted and p_start is first)
        if p.timestamp < p_start.timestamp:
             # Sanity fallback, dist to start
            d_lat = p.lat - p_start.lat
            d_lon = p.lon - p_start.lon
            avg_lat = math.radians((p.lat + p_start.lat) / 2.0)
            d_lat_m = d_lat * 111320.0
            d_lon_m = d_lon * 111320.0 * math.cos(avg_lat)
            dist = math.sqrt(d_lat_m*d_lat_m + d_lon_m*d_lon_m)
            sed_errors.append(dist)
            continue
            
        error = calculate_sed_error(p, p_start, p_end)
        sed_errors.append(error)
        
    if not sed_errors:
        return {'average_sed': 0.0, 'max_sed': 0.0, 'rmse': 0.0, 'sed_errors': []}
        
    avg_sed = sum(sed_errors) / len(sed_errors)
    max_sed = max(sed_errors)
    mse = sum(e*e for e in sed_errors) / len(sed_errors)
    rmse = math.sqrt(mse)
    
    return {
        'average_sed': avg_sed,
        'max_sed': max_sed,
        'rmse': rmse,
        'sed_errors': sed_errors
    }
