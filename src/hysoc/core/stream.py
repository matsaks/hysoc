import pandas as pd 
from typing import Iterator, Dict, Optional
from pathlib import Path
from .point import Point

class TrajectoryStream:
    """
    Simulates a live GPS stream by reading a CSV file line-by-line.
    Handles both multi-object streams (via column) and single-object files (via default_id)
    """
    def __init__(
        self,
        filepath: str | Path,
        sep: str= ',',
        col_mapping: Dict[str, str] = None,
        default_obj_id: Optional[str] = 'unknown_obj'
    ):
        self.filepath = Path(filepath)
        self.sep = sep
        self.default_obj_id = default_obj_id

        self.mapping = col_mapping or {
            'lat': 'lat',
            'lon': 'lon',
            'timestamp': 'timestamp',
            'obj_id': 'obj_id',
            'road_id': 'osm_way_id'
        }

    def stream(self) -> Iterator[Point]:
        """
        Yields points from the stream one by one.
        """
        header = pd.read_csv(self.filepath, nrows=0, sep=self.sep)
        header_cols = set(header.columns)
        
        # Auto-detect columns if the configured ones are missing
        variants = {
            'lat': ['lat', 'latitude', 'y'],
            'lon': ['lon', 'longitude', 'x'],
            'timestamp': ['timestamp', 'time', 'datetime', 't', 'date'],
            'obj_id': ['obj_id', 'oid', 'user_id', 'trajectory_id'],
            'road_id': ['road_id', 'osm_way_id', 'edge_id']
        }
        
        for key, possible_names in variants.items():
            current_col = self.mapping.get(key)
            # If current mapping is invalid/missing in file, try to find an alternative
            if current_col not in header_cols:
                for name in possible_names:
                    if name in header_cols:
                        self.mapping[key] = name
                        break
        
        has_id_col = self.mapping.get('obj_id') in header_cols
        has_road_col = self.mapping.get('road_id') in header_cols

        with pd.read_csv(self.filepath, chunksize=1000, sep=self.sep) as reader:
            for chunk in reader:
                chunk[self.mapping['timestamp']] = pd.to_datetime(chunk[self.mapping['timestamp']])
                
                for _, row in chunk.iterrows():
                    yield Point(
                        lat=row[self.mapping['lat']],
                        lon=row[self.mapping['lon']],
                        timestamp=row[self.mapping['timestamp']],
                        obj_id=row[self.mapping['obj_id']] if has_id_col else self.default_obj_id,
                        road_id=row[self.mapping['road_id']] if has_road_col else None
                    )

