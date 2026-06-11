"""TRACE network-semantic referential move compressor (HYSOC-N Module III)."""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import Any

from constants.geo_defaults import EARTH_RADIUS_M
from core.compression import SegmentResult
from core.point import Point
from core.trace_config import TraceConfig

_BYTES_LITERAL: int = 4
_BYTES_MATCH_BASE: int = 12
_BYTES_MISMATCH: int = 4
_BYTES_SEGMENT_HEADER: int = 16


@dataclass
class Reference:
    """A stored reference trajectory in the shared index H."""

    ref_id: int
    e_seq: list[int]
    v_seq: list[int]
    last_access_call: int


class TraceCompressor:
    """TRACE move-segment compressor with a persistent shared reference index."""

    def __init__(self, config: TraceConfig = TraceConfig()):
        self.config = config
        self.references: dict[int, Reference] = {}
        self._ref_id_counter: int = 0
        self._compress_call_count: int = 0
        # Inverted k-mer index: hash(seq_type, kmer) -> [(ref_id, offset, seq_type)].
        self.kmer_index: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
        self.diagnostics: dict[str, Any] = {
            "compress_calls": 0,
            "input_points": 0,
            "retained_points": 0,
            "retention_ratio": 0.0,
            "compress_total_time_s": 0.0,
            "speed_rep_time_s": 0.0,
            "referential_time_s": 0.0,
            "reference_manage_time_s": 0.0,
            "reference_delete_time_s": 0.0,
            "literal_factors_e": 0,
            "match_factors_e": 0,
            "literal_factors_v": 0,
            "match_factors_v": 0,
            "references_count": 0,
            "kmer_bucket_count": 0,
            "kmer_entry_count": 0,
        }

    def compress(self, points: list[Point]) -> SegmentResult:
        """Compress a Move segment and return a SegmentResult."""
        if not points:
            return SegmentResult(
                kind="move",
                start_time=None,
                end_time=None,
                keypoints=[],
                encoded_bytes=0,
            )

        self._compress_call_count += 1
        self.diagnostics["compress_calls"] += 1
        self.diagnostics["input_points"] += len(points)
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        retained, speed_rep = self._speed_based_representation(points)
        self.diagnostics["speed_rep_time_s"] += time.perf_counter() - t0
        self.diagnostics["retained_points"] += len(retained)

        t0 = time.perf_counter()
        e_factors, v_factors = self._referential_compression(speed_rep)
        self.diagnostics["referential_time_s"] += time.perf_counter() - t0

        used_refs = _extract_used_refs(e_factors, v_factors)
        t0 = time.perf_counter()
        self._manage_references(speed_rep, used_refs, self._compress_call_count)
        self.diagnostics["reference_manage_time_s"] += time.perf_counter() - t0

        n_in = len(points)
        n_ret = len(retained)
        self.diagnostics["literal_factors_e"] = sum(
            1 for f in e_factors if not isinstance(f, tuple)
        )
        self.diagnostics["match_factors_e"] = sum(
            1 for f in e_factors if isinstance(f, tuple)
        )
        self.diagnostics["literal_factors_v"] = sum(
            1 for f in v_factors if not isinstance(f, tuple)
        )
        self.diagnostics["match_factors_v"] = sum(
            1 for f in v_factors if isinstance(f, tuple)
        )
        self.diagnostics["references_count"] = len(self.references)
        self.diagnostics["kmer_bucket_count"] = len(self.kmer_index)
        self.diagnostics["kmer_entry_count"] = sum(
            len(v) for v in self.kmer_index.values()
        )
        self.diagnostics["retention_ratio"] = n_ret / n_in if n_in > 0 else 0.0
        self.diagnostics["compress_total_time_s"] += time.perf_counter() - t_total

        return SegmentResult(
            kind="move",
            start_time=points[0].timestamp,
            end_time=points[-1].timestamp,
            keypoints=retained,
            encoded_bytes=_encoded_bytes(e_factors, v_factors),
            e_factors=e_factors,
            v_factors=v_factors,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostics)

    def _speed_based_representation(
        self, points: list[Point]
    ) -> tuple[list[Point], list[tuple]]:
        """Retain points where road_id changes or speed deviates beyond gamma."""
        retained: list[Point] = []
        speed_rep: list[tuple] = []

        if not points:
            return retained, speed_rep

        current_road: int | str | None = None
        seg_offset: float = 0.0
        last_speed: float = 0.0

        for i, p in enumerate(points):
            if i == 0:
                current_road = p.road_id
                seg_offset = 0.0
                last_speed = 0.0
                retained.append(p)
                speed_rep.append((current_road, 1, 0.0, 0.0))
                continue

            prev = points[i - 1]
            dist = _haversine(prev, p)
            dt = (p.timestamp - prev.timestamp).total_seconds()
            speed = dist / dt if dt > 0 else last_speed

            if p.road_id != current_road:
                # Carry the real entry speed forward, not an artificial zero.
                current_road = p.road_id
                seg_offset = 0.0
                last_speed = speed
                retained.append(p)
                speed_rep.append((current_road, 1, 0.0, speed))
                continue

            seg_offset += dist

            if abs(speed - last_speed) > self.config.gamma:
                last_speed = speed
                retained.append(p)
                speed_rep.append((current_road, 1, seg_offset, speed))

        return retained, speed_rep

    def _referential_compression(
        self, speed_rep: list[tuple]
    ) -> tuple[list, list]:
        """Compress the road-id and quantised-speed sequences via k-mer matching."""
        if not speed_rep:
            return [], []

        e_seq = [entry[0] for entry in speed_rep]

        eta = self.config.epsilon
        v_seq = [
            round(entry[3] / (0.5 * eta)) if eta > 0 else int(entry[3])
            for entry in speed_rep
        ]

        return (
            self._compress_sequence(e_seq, "E"),
            self._compress_sequence(v_seq, "V"),
        )

    def _compress_sequence(self, sequence: list, seq_type: str) -> list:
        """LZ-style k-mer compression of a single sequence."""
        compressed = []
        n = len(sequence)
        k = self.config.k
        i = 0

        while i < n:
            if i + k > n:
                # Fewer than k elements remain.
                compressed.append(sequence[i])
                i += 1
                continue

            kmer = tuple(sequence[i : i + k])
            kmer_hash = hash((seq_type, kmer))
            candidates = self.kmer_index.get(kmer_hash, [])

            best: tuple | None = None
            best_len = -1

            for ref_id, ref_offset, r_type in candidates:
                if r_type != seq_type:
                    continue
                ref = self.references.get(ref_id)
                if ref is None:
                    continue

                if seq_type == "E":
                    ref_seq = ref.e_seq
                else:
                    ref_seq = ref.v_seq

                # Verify k-mer to guard against hash collisions.
                if len(ref_seq) < ref_offset + k:
                    continue
                if tuple(ref_seq[ref_offset : ref_offset + k]) != kmer:
                    continue

                match_len = k
                mi = i + k
                ri = ref_offset + k
                mismatch = None

                while mi < n and ri < len(ref_seq):
                    if sequence[mi] == ref_seq[ri]:
                        match_len += 1
                        mi += 1
                        ri += 1
                    else:
                        mismatch = sequence[mi]
                        break

                if match_len > best_len:
                    best_len = match_len
                    best = (ref_id, ref_offset, match_len, mismatch)

            if best is not None:
                compressed.append(best)
                _, _, length, mismatch = best
                # Consume the trailing mismatch element when one was recorded.
                i += length + (1 if mismatch is not None else 0)
            else:
                compressed.append(sequence[i])
                i += 1

        return compressed

    def _manage_references(
        self,
        speed_rep: list[tuple],
        used_refs: set[int],
        current_call: int,
    ) -> None:
        """Refresh accessed references, add current segment to H, prune stale ones."""
        for ref_id in used_refs:
            ref = self.references.get(ref_id)
            if ref is not None:
                ref.last_access_call = current_call

        if speed_rep:
            e_seq = [entry[0] for entry in speed_rep]
            eta = self.config.epsilon
            v_seq = [
                round(entry[3] / (0.5 * eta)) if eta > 0 else int(entry[3])
                for entry in speed_rep
            ]
            self._ref_id_counter += 1
            new_ref = Reference(
                ref_id=self._ref_id_counter,
                e_seq=e_seq,
                v_seq=v_seq,
                last_access_call=current_call,
            )
            self.references[self._ref_id_counter] = new_ref
            self._update_kmer_index(new_ref)

        self._reference_deletion(current_call)

    def _reference_deletion(self, current_call: int) -> None:
        """Remove references whose freshness falls below the cleanup threshold."""
        t0 = time.perf_counter()

        if not self.references:
            self.diagnostics["reference_delete_time_s"] += time.perf_counter() - t0
            return

        decay = self.config.decay_lambda
        threshold_c = self.config.cleanup_threshold

        scores: dict[int, float] = {
            ref.ref_id: decay ** max(0, current_call - ref.last_access_call)
            for ref in self.references.values()
        }
        avg = sum(scores.values()) / len(scores)
        cutoff = threshold_c * avg

        for ref_id, score in scores.items():
            if score < cutoff:
                self._delete_reference(ref_id)

        self.diagnostics["reference_delete_time_s"] += time.perf_counter() - t0

    def _delete_reference(self, ref_id: int) -> None:
        """Remove a reference from H and purge its entries from the k-mer index."""
        if ref_id not in self.references:
            return
        del self.references[ref_id]
        empty_buckets = []
        for h, entries in self.kmer_index.items():
            filtered = [e for e in entries if e[0] != ref_id]
            if filtered:
                self.kmer_index[h] = filtered
            else:
                empty_buckets.append(h)
        for h in empty_buckets:
            del self.kmer_index[h]

    def _update_kmer_index(self, ref: Reference) -> None:
        """Index all k-mers from E and V sequences of a new reference."""
        k = self.config.k
        for seq_type, seq in (("E", ref.e_seq), ("V", ref.v_seq)):
            for i in range(len(seq) - k + 1):
                kmer = tuple(seq[i : i + k])
                kmer_hash = hash((seq_type, kmer))
                self.kmer_index[kmer_hash].append((ref.ref_id, i, seq_type))


def _haversine(p1: Point, p2: Point) -> float:
    """Equirectangular distance approximation in metres."""
    lat1 = math.radians(p1.lat)
    lat2 = math.radians(p2.lat)
    dlat = lat2 - lat1
    dlon = math.radians(p2.lon - p1.lon)
    x = dlon * math.cos((lat1 + lat2) / 2.0)
    return EARTH_RADIUS_M * math.sqrt(x * x + dlat * dlat)


def _extract_used_refs(e_factors: list, v_factors: list) -> set[int]:
    """Collect all reference IDs cited in E and V factor lists."""
    used: set[int] = set()
    for factors in (e_factors, v_factors):
        for f in factors:
            if isinstance(f, tuple) and len(f) == 4:
                used.add(f[0])
    return used


def _encoded_bytes(e_factors: list, v_factors: list) -> int:
    """Byte cost of the TRACE-encoded segment."""
    total = _BYTES_SEGMENT_HEADER
    for factors in (e_factors, v_factors):
        for f in factors:
            if isinstance(f, tuple):
                total += _BYTES_MATCH_BASE + (
                    _BYTES_MISMATCH if f[3] is not None else 0
                )
            else:
                total += _BYTES_LITERAL
    return total
