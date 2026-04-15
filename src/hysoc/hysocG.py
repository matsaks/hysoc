"""
HYSOC: Hybrid Trajectory Compression Framework — Main Orchestrator

This module is the central entry point for the HYSOC pipeline.
It orchestrates the full compression workflow by delegating to:
  - Module I:    Streaming Segmentation (STEP)         → segmentation/step.py
  - Module II:   Stop Compression (StopCompressor)     → stop_compression/compressor.py
  - Module III-A: Move Compression — Geometric (SQUISH+DP) → move_compression/squish.py + dp.py
  - Module III-B: Move Compression — Network-Semantic   → move_compression/trace.py

Run directly:
    python -m hysoc.modules.hysoc
"""

import os
import math
import time
from typing import List, Optional

from core.point import Point
from core.segment import Segment, Stop, Move
from core.compression import (
    CompressionStrategy,
    HYSOCConfig,
    CompressedSegment,
    CompressedTrajectory,
)
from engines.segmentation.step import STEPSegmenter
from engines.stop_compression.compressor import StopCompressor
from engines.move_compression.squish import SquishCompressor
from engines.move_compression.dp import DouglasPeuckerCompressor
from engines.move_compression.trace import TraceCompressor
from engines.map_matching.matcher import OnlineMapMatcher
from constants.geo_defaults import EARTH_RADIUS_M

# ---------------------------------------------------------------------------
# Module-level constant for the default demo input file (not in constants/).
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FILE: str = os.path.join("data", "raw", "subset_50", "4494499.csv")


class HYSOCGCompressor:
    """
    Unified HYSOC trajectory compression orchestrator.

    Acts as a true streaming pipeline.  As points flow in via ``process_point``,
    they get map-matched, segmented, and then compressed block-by-block.
    Memory is kept low as it does not perpetually buffer history.
    """

    def __init__(self, config: HYSOCConfig = None):
        self.config = config if config is not None else HYSOCConfig()

        if self.config.move_compression_strategy == CompressionStrategy.NETWORK_SEMANTIC:
            if self.config.enable_map_matching and self.config.osm_graph is None:
                raise ValueError(
                    "NETWORK_SEMANTIC strategy with map matching requires osm_graph."
                )

        # Module I: Segmentation
        self.segmenter = STEPSegmenter(
            max_eps=self.config.stop_max_eps_meters,
            min_duration_seconds=self.config.stop_min_duration_seconds,
        )

        # Module II: Stop Compression
        self.stop_compressor = StopCompressor()

        # Module III: Move Compression
        if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
            self.squish_compressor = SquishCompressor(
                capacity=self.config.squish_buffer_capacity
            )
            self.dp_compressor = DouglasPeuckerCompressor(
                epsilon_meters=self.config.dp_epsilon_meters
            )
        else:  # NETWORK_SEMANTIC
            self.move_compressor = TraceCompressor(config=self.config.trace_config)

        # Optional map matcher
        self.map_matcher: Optional[OnlineMapMatcher] = None
        if self.config.enable_map_matching and self.config.osm_graph is not None:
            self.map_matcher = OnlineMapMatcher(self.config.osm_graph)

        # Metrics
        self._total_points_in = 0
        self._total_points_compressed = 0
        self.diagnostics = {
            "map_matching_time_s": 0.0,
            "segmentation_time_s": 0.0,
            "compression_time_s": 0.0,
            "trace_time_s": 0.0,
            "trace_retained_extract_time_s": 0.0,
            "retention_input_points": 0,
            "retention_kept_points": 0,
            "retention_kept_road_change": 0,
            "retention_kept_speed_change": 0,
            "retention_forced_last_point": 0,
            "retention_move_segments": 0,
        }

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def process_point(self, point: Point) -> List[CompressedSegment]:
        """
        Processes a single point through the streaming pipeline.
        Returns any fully compressed segments that were closed by this point.
        """
        self._total_points_in += 1

        # Stage 1: Map Matching
        if self.map_matcher is not None:
            t0 = time.perf_counter()
            matched_point = self.map_matcher.process_point(point)
            t1 = time.perf_counter()
            self.diagnostics["map_matching_time_s"] += float(t1 - t0)
            if matched_point is None:
                return []
            point = matched_point

        # Stage 2: Segmentation
        t0 = time.perf_counter()
        segments = self.segmenter.process_point(point)
        t1 = time.perf_counter()
        self.diagnostics["segmentation_time_s"] += float(t1 - t0)

        # Stage 3: Compression
        compressed = []
        t0 = time.perf_counter()
        for seg in segments:
            c_seg = self._compress_segment(seg)
            if c_seg is not None:
                compressed.append(c_seg)
        t1 = time.perf_counter()
        self.diagnostics["compression_time_s"] += float(t1 - t0)

        return compressed

    def flush(self) -> List[CompressedSegment]:
        """Flushes and compresses any remaining buffered segments."""
        compressed = []

        # Flush map matcher buffers through segmenter
        if self.map_matcher is not None:
            for point in self.map_matcher.flush():
                t0 = time.perf_counter()
                segments = self.segmenter.process_point(point)
                t1 = time.perf_counter()
                self.diagnostics["segmentation_time_s"] += float(t1 - t0)
                t0 = time.perf_counter()
                for seg in segments:
                    c_seg = self._compress_segment(seg)
                    if c_seg is not None:
                        compressed.append(c_seg)
                t1 = time.perf_counter()
                self.diagnostics["compression_time_s"] += float(t1 - t0)

        # Flush segmenter
        t0 = time.perf_counter()
        for seg in self.segmenter.flush():
            c_seg = self._compress_segment(seg)
            if c_seg is not None:
                compressed.append(c_seg)
        t1 = time.perf_counter()
        self.diagnostics["compression_time_s"] += float(t1 - t0)

        return compressed

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def compress(self, points: List[Point]) -> CompressedTrajectory:
        """
        Batch wrapper — pushes all points through the streaming pipeline
        and collects the result.
        """
        self.__init__(self.config)

        compressed_segments = []
        for point in points:
            compressed_segments.extend(self.process_point(point))
        compressed_segments.extend(self.flush())

        factor = 0.0
        if self._total_points_compressed > 0:
            factor = self._total_points_in / self._total_points_compressed

        return CompressedTrajectory(
            original_points=points,
            compressed_segments=compressed_segments,
            total_original_points=self._total_points_in,
            total_compressed_points=self._total_points_compressed,
            overall_compression_factor=factor,
            compression_strategy=self.config.move_compression_strategy,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_segment(self, seg: Segment) -> Optional[CompressedSegment]:
        """Routes a detected segment to the appropriate compressor."""
        if isinstance(seg, Stop):
            if self.config.compress_stops:
                compressed_data = self.stop_compressor.compress(seg.points)
                total_compressed = 1
            else:
                compressed_data = seg
                total_compressed = len(seg.points)

            compression_factor = (
                len(seg.points) / total_compressed
                if total_compressed > 0
                else 0.0
            )
            self._total_points_compressed += total_compressed

            return CompressedSegment(
                segment_type="stop",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_factor=compression_factor,
            )

        elif isinstance(seg, Move):
            if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
                squish_result = self.squish_compressor.compress(seg.points)
                compressed_data = self.dp_compressor.compress(squish_result)
                total_compressed = len(compressed_data)
            else:
                t0 = time.perf_counter()
                trace_result = self.move_compressor.compress(seg.points)
                t1 = time.perf_counter()
                self.diagnostics["trace_time_s"] += float(t1 - t0)
                t0 = time.perf_counter()
                retained_points, counters = self._extract_retained_points_from_trace(seg.points)
                t1 = time.perf_counter()
                self.diagnostics["trace_retained_extract_time_s"] += float(t1 - t0)
                self.diagnostics["retention_move_segments"] += 1
                self.diagnostics["retention_input_points"] += counters["input_points"]
                self.diagnostics["retention_kept_points"] += counters["kept_points"]
                self.diagnostics["retention_kept_road_change"] += counters["road_change_kept"]
                self.diagnostics["retention_kept_speed_change"] += counters["speed_change_kept"]
                self.diagnostics["retention_forced_last_point"] += counters["forced_last_point"]
                compressed_data = {
                    "trace_result": trace_result,
                    "retained_points": retained_points,
                }
                total_compressed = len(retained_points)

            compression_factor = (
                len(seg.points) / total_compressed
                if total_compressed > 0
                else 0.0
            )
            self._total_points_compressed += total_compressed

            return CompressedSegment(
                segment_type="move",
                original_segment=seg,
                compressed_data=compressed_data,
                compression_factor=compression_factor,
            )

        return None

    def _extract_retained_points_from_trace(self, points: List[Point]) -> tuple[List[Point], dict]:
        """Identifies retained velocity-change points for TRACE rendering."""
        if not points or len(points) < 2:
            n = len(points) if points else 0
            counters = {
                "input_points": n,
                "kept_points": n,
                "road_change_kept": n,
                "speed_change_kept": 0,
                "forced_last_point": 0,
            }
            return points, counters

        retained_points = []
        gamma = self.config.trace_config.gamma
        road_change_kept = 0
        speed_change_kept = 0
        forced_last_point = 0

        def lat_lon_dist(p1: Point, p2: Point) -> float:
            lat1 = math.radians(p1.lat)
            lat2 = math.radians(p2.lat)
            dlat = lat2 - lat1
            dlon = math.radians(p2.lon - p1.lon)
            x = dlon * math.cos((lat1 + lat2) / 2.0)
            y = dlat
            return EARTH_RADIUS_M * math.sqrt(x * x + y * y)

        current_road_id = None
        last_stored_speed = -1.0

        for i, p in enumerate(points):
            if i == 0 or p.road_id != current_road_id:
                current_road_id = p.road_id
                retained_points.append(p)
                road_change_kept += 1
                last_stored_speed = 0.0
                continue

            prev_p = points[i - 1]
            dist = lat_lon_dist(prev_p, p)
            time_diff = (p.timestamp - prev_p.timestamp).total_seconds()

            if time_diff > 0:
                current_speed = dist / time_diff
            else:
                current_speed = last_stored_speed

            if abs(current_speed - last_stored_speed) > gamma:
                retained_points.append(p)
                speed_change_kept += 1
                last_stored_speed = current_speed

        if retained_points and retained_points[-1] != points[-1]:
            retained_points.append(points[-1])
            forced_last_point = 1

        counters = {
            "input_points": len(points),
            "kept_points": len(retained_points),
            "road_change_kept": road_change_kept,
            "speed_change_kept": speed_change_kept,
            "forced_last_point": forced_last_point,
        }
        return retained_points, counters

    def get_diagnostics(self) -> dict:
        diag = dict(self.diagnostics)
        if self.map_matcher is not None and hasattr(self.map_matcher, "get_diagnostics"):
            diag["map_matcher"] = self.map_matcher.get_diagnostics()
        if (
            self.config.move_compression_strategy == CompressionStrategy.NETWORK_SEMANTIC
            and hasattr(self.move_compressor, "get_diagnostics")
        ):
            diag["trace"] = self.move_compressor.get_diagnostics()
        if diag["retention_input_points"] > 0:
            diag["retention_ratio"] = float(
                diag["retention_kept_points"] / diag["retention_input_points"]
            )
        else:
            diag["retention_ratio"] = 0.0
        return diag

    def get_compression_summary(self) -> str:
        """Returns a human-readable summary of the current configuration."""
        lines = [
            "=" * 60,
            "HYSOC Compression Configuration",
            "=" * 60,
            f"Segmentation: STEP (ε={self.config.stop_max_eps_meters}m, "
            f"T={self.config.stop_min_duration_seconds}s)",
            f"Stop Compression: {'Enabled' if self.config.compress_stops else 'Disabled'}",
            f"Move Compression: {self.config.move_compression_strategy.value.upper()}",
        ]

        if self.config.move_compression_strategy == CompressionStrategy.GEOMETRIC:
            lines.append(
                f"  - SquishCompressor (capacity={self.config.squish_buffer_capacity}) → "
                f"DouglasPeuckerCompressor (ε={self.config.dp_epsilon_meters}m)"
            )
        else:
            lines.append("  - TraceCompressor (Network-Semantic)")
            lines.append(
                f"  - Map Matching: {'Enabled' if self.config.enable_map_matching else 'Disabled'}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Main entry point
# ======================================================================

def main(input_file: Optional[str] = None):
    """Loads the default trajectory file and runs the full HYSOC pipeline."""
    from core.stream import TrajectoryStream

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if input_file:
        data_path = os.path.abspath(input_file)
    else:
        data_path = os.path.join(project_root, DEFAULT_INPUT_FILE)

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    stream = TrajectoryStream(
        filepath=data_path,
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
    )
    points = list(stream.stream())

    config = HYSOCConfig(
        move_compression_strategy=CompressionStrategy.GEOMETRIC,
    )
    compressor = HYSOCGCompressor(config=config)
    result = compressor.compress(points)

    print(f"Compressed {result.total_original_points} → {result.total_compressed_points} points "
          f"({len(result.compressed_segments)} segments)")


if __name__ == "__main__":
    main()


# Backward-compatible alias used by existing scripts.
HYSOCCompressor = HYSOCGCompressor

