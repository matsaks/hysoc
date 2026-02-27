from collections import deque
from typing import List, Optional
import networkx as nx
import copy

import dataclasses
from hysoc.core.point import Point
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap


class OnlineMapMatcher:
    """
    Online Map Matcher using Hidden Markov Model (HMM) via LeuvenMapMatching.
    It processes points one by one, maintaining a sliding window to provide future
    context for map-matching older points. 
    
    This matches the requirement for online streaming where the full trajectory
    is not known in advance.
    """

    def __init__(
        self, 
        G: nx.MultiDiGraph, 
        window_size: int = 15, 
        max_dist: float = 50.0,
        max_dist_init: float = 100.0,
        min_prob_norm: float = 0.001
    ):
        """
        Args:
            G: The osmnx graph to match against. Must be unprojected (Lat/Lon EPSG:4326).
            window_size: How many points to keep in the buffer for HMM context.
                         Larger window = more accuracy, but higher latency.
            max_dist: Maximum distance from point to edge in meters.
            max_dist_init: Maximum distance for the first point of a sequence.
            min_prob_norm: Minimum normalized probability for a matched path.
        """
        self.G = G
        self.window_size = window_size
        self.max_dist = max_dist
        self.max_dist_init = max_dist_init
        self.min_prob_norm = min_prob_norm
        
        self.buffer: deque[Point] = deque()
        
        # Initialize InMemMap for LeuvenMapMatching
        self.map_con = InMemMap("network", use_latlon=True)
        self._build_map()

    def _build_map(self):
        """Converts the OSMnx graph to LeuvenMapMatching's internal representation."""
        # Add nodes
        for node_id, data in self.G.nodes(data=True):
            # LeuvenMapMatching expects Lat/Lon. OSMnx nodes store it as y/x.
            self.map_con.add_node(node_id, (data['y'], data['x']))
            
        # Add edges (handling bidirectional routing if MultiDiGraph)
        for u, v, _ in self.G.edges(data=True):
            self.map_con.add_edge(u, v)

    def process_point(self, point: Point) -> Optional[Point]:
        """
        Ingest a new GPS point. If the sliding window is full, map-matches the 
        entire window using Viterbi and yields the oldest point with its resolved
        'road_id'.
        
        Args:
            point: The incoming GPS Point.
            
        Returns:
            The oldest Point in the window with 'road_id' set, or None if the buffer
            is not yet full.
        """
        self.buffer.append(point)
        
        # Wait until we have enough context
        if len(self.buffer) < self.window_size:
            return None
            
        # Buffer is full, run matching on the whole window
        matched_point = self._match_window()
        
        # Pop the oldest point to make room for the next one
        self.buffer.popleft()
        return matched_point

    def flush(self) -> List[Point]:
        """
        Flushes all remaining points in the sliding window at the end of the stream,
        assigning them road_ids based on the best matching of the shrinking tail.
        """
        flushed_points = []
        while self.buffer:
            matched_point = self._match_window()
            if matched_point:
                flushed_points.append(matched_point)
            else:
                # If matching completely fails, just yield the raw point
                flushed_points.append(self.buffer[0])
            self.buffer.popleft()
            
        return flushed_points

    def _match_window(self) -> Optional[Point]:
        """
        Matches the current window and assigns road_id to the oldest point (index 0).
        """
        if not self.buffer:
            return None
            
        # Create a deep copy of the oldest point so we don't mutate the original
        # immediately if it's referenced elsewhere.
        oldest_point = copy.deepcopy(self.buffer[0])
            
        # Extract lat/lon path. We pass (lat, lon) to matcher.
        path = [(p.lat, p.lon) for p in self.buffer]
        
        matcher = DistanceMatcher(
            self.map_con, 
            max_dist=self.max_dist, 
            max_dist_init=self.max_dist_init, 
            min_prob_norm=self.min_prob_norm
        )
        
        try:
            states, _ = matcher.match(path)
        except Exception:
            states = []
            
        # states is a list of node tuples representing edges: [(u, v), (v, w), ...]
        # Note: LMM sometimes returns fewer states than input points if it drops noisy points,
        # but the states sequence represents the best semantic route.
        # We index 0 to get the matched edge for the oldest point.
        if states and len(states) > 0:
            u, v = states[0]
            
            # Extract actual osm_way_id from the original graph edges
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                # MultiDiGraph returns a dict of keys. Usually 0 is the primary edge.
                if 0 in edge_data and 'osmid' in edge_data[0]:
                    osmid = edge_data[0]['osmid']
                    # osmid can be a list if multple ways merged
                    r_id = str(osmid[0]) if isinstance(osmid, list) else str(osmid)
                else:
                    r_id = f"{u}-{v}"
                    print(f"DEBUG: missing osmid at 0. keys: {list(edge_data.keys())}, data: {edge_data}")
            else:
                r_id = f"{u}-{v}"
                print(f"DEBUG: get_edge_data({u}, {v}) returned None!")
            
            # Snap point to geometry
            snapped_lat, snapped_lon = oldest_point.lat, oldest_point.lon # default
            if edge_data and 0 in edge_data:
                from shapely.geometry import Point as ShapelyPoint, LineString
                raw_pt = ShapelyPoint(oldest_point.lon, oldest_point.lat)
                if 'geometry' in edge_data[0]:
                    geom = edge_data[0]['geometry']
                else:
                    u_node = self.G.nodes[u]
                    v_node = self.G.nodes[v]
                    geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
                
                # Project GPS point onto matched edge geometry using Shapely
                proj_dist = geom.project(raw_pt)
                snapped_pt = geom.interpolate(proj_dist)
                snapped_lon, snapped_lat = snapped_pt.x, snapped_pt.y

            # Use dataclasses.replace to bypass frozen instance restrictions
            oldest_point = dataclasses.replace(
                oldest_point, 
                road_id=r_id,
                lat=snapped_lat,
                lon=snapped_lon
            )
                
        return oldest_point
