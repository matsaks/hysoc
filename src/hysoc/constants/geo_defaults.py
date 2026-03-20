"""Geographic / Earth-related constants."""

from __future__ import annotations

# Mean Earth radius in meters (WGS84-ish approximation).
EARTH_RADIUS_M: float = 6371000.0

# Approximate meters per degree of latitude (equatorial; used for fast local
# planar projections where haversine precision is unnecessary).
METERS_PER_DEGREE_LAT: float = 111320.0

