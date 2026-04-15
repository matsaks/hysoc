from typing import List, Dict, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from hysoc.core.point import Point
import collections
import math
import time
from hysoc.constants.geo_defaults import EARTH_RADIUS_M
from hysoc.constants.trace_defaults import (
    TRACE_GAMMA,
    TRACE_EPSILON,
    TRACE_K,
    TRACE_ALPHA,
    TRACE_CLEANUP_THRESHOLD,
    TRACE_DECAY_LAMBDA,
)

@dataclass
class TraceConfig:
    """Configuration for TRACE compressor."""
    gamma: float = TRACE_GAMMA  # Speed threshold
    epsilon: float = TRACE_EPSILON  # Error bound for prediction
    k: int = TRACE_K  # k-mer length
    alpha: int = TRACE_ALPHA  # Threshold for reference rewriting
    cleanup_threshold: float = TRACE_CLEANUP_THRESHOLD  # Threshold C for reference deletion
    decay_lambda: float = TRACE_DECAY_LAMBDA  # Decay factor for freshness

@dataclass
class Reference:
    """Represents a reference trajectory."""
    ref_id: int
    points: List[Point]
    # Stored sequences for referential compression lookup
    e_seq: List[int] = field(default_factory=list) # Road IDs sequence
    v_seq: List[float] = field(default_factory=list) # Speed sequence
    
    # Factor matrix FA and rp as described in Algorithm 2
    factor_matrix: Dict[Tuple[int, int], int] = field(default_factory=dict)
    rp: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict) # index -> list of (M, count)
    freshness: float = 0.0
    last_access_time: float = 0.0

class TraceCompressor:
    """
    Implementation of TRACE: Real-time Compression of Streaming Trajectories in Road Networks.
    
    This module implements Module III (Part B) of the HYSOC framework.
    It provides network-constrained move compression using:
    1. Speed-based representation (Section 3.1 in trace.txt)
    2. Referential compression (Section 3.2 in trace.txt)
    3. Reference management (Selection, Deletion, Rewriting) (Section 3.3 in trace.txt)
    """

    def __init__(self, config: TraceConfig = TraceConfig()):
        self.config = config
        self.references: Dict[int, Reference] = {} # ref_id -> Reference
        self.reference_freshness_sum: float = 0.0
        self.current_ref_id_counter: int = 0
        
        # Inverted index for k-mer matching: k-mer hash -> list of (ref_id, offset, type)
        # Type is 'E' or 'V'
        self.kmer_index: Dict[int, List[Tuple[int, int, str]]] = collections.defaultdict(list)
        self.diagnostics: Dict[str, Any] = {
            "compress_calls": 0,
            "input_points": 0,
            "speed_rep_points": 0,
            "speed_rep_reduction_ratio": 0.0,
            "compress_total_time_s": 0.0,
            "speed_rep_time_s": 0.0,
            "referential_time_s": 0.0,
            "reference_manage_time_s": 0.0,
            "reference_delete_time_s": 0.0,
            "factor_count_e": 0,
            "factor_count_v": 0,
            "tuple_match_factors_e": 0,
            "tuple_match_factors_v": 0,
            "literal_factors_e": 0,
            "literal_factors_v": 0,
            "references_count": 0,
            "kmer_bucket_count": 0,
            "kmer_entry_count": 0,
        }

    def compress(self, points: List[Point]) -> Any:
        """
        Main entry point for compressing a sequence of points (a Move segment).
        
        Args:
            points: List of GPS points (assumed to be map-matched with road_id).
            
        Returns:
            A compact representation of the trajectory.
        """
        if not points:
            return None
        self.diagnostics["compress_calls"] += 1
        self.diagnostics["input_points"] += len(points)
        t_total_0 = time.perf_counter()

        # Step 1: Speed-based Representation
        # Convert raw points to [(road_id, direction, offset, speed), ...]
        t0 = time.perf_counter()
        speed_rep = self._speed_based_representation(points)
        t1 = time.perf_counter()
        self.diagnostics["speed_rep_time_s"] += float(t1 - t0)
        self.diagnostics["speed_rep_points"] += len(speed_rep)
        if len(points) > 0:
            self.diagnostics["speed_rep_reduction_ratio"] = float(len(speed_rep) / len(points))

        # Step 2: Referential Compression
        # Replace subsequences with references to existing trajectories
        t0 = time.perf_counter()
        compressed_rep = self._referential_compression(speed_rep)
        t1 = time.perf_counter()
        self.diagnostics["referential_time_s"] += float(t1 - t0)

        # Step 3: Reference Management (Selection, Deletion, Rewriting)
        # Updates the reference set based on usage
        current_time = points[-1].timestamp.timestamp() # Use last point time as approximation
        
        # Identify used references to update their freshness
        used_refs = set()
        for key in ['E', 'V']:
            for item in compressed_rep.get(key, []):
                if isinstance(item, tuple) and len(item) >= 1:
                    # Item is (ref_id, start, len, mismatch)
                    used_refs.add(item[0])

        t0 = time.perf_counter()
        self._manage_references(points, speed_rep, used_refs, current_time)
        t1 = time.perf_counter()
        self.diagnostics["reference_manage_time_s"] += float(t1 - t0)

        self.diagnostics["factor_count_e"] = len(compressed_rep.get("E", []))
        self.diagnostics["factor_count_v"] = len(compressed_rep.get("V", []))
        self.diagnostics["tuple_match_factors_e"] = sum(
            1 for item in compressed_rep.get("E", []) if isinstance(item, tuple)
        )
        self.diagnostics["tuple_match_factors_v"] = sum(
            1 for item in compressed_rep.get("V", []) if isinstance(item, tuple)
        )
        self.diagnostics["literal_factors_e"] = (
            self.diagnostics["factor_count_e"] - self.diagnostics["tuple_match_factors_e"]
        )
        self.diagnostics["literal_factors_v"] = (
            self.diagnostics["factor_count_v"] - self.diagnostics["tuple_match_factors_v"]
        )
        self.diagnostics["references_count"] = len(self.references)
        self.diagnostics["kmer_bucket_count"] = len(self.kmer_index)
        self.diagnostics["kmer_entry_count"] = sum(len(v) for v in self.kmer_index.values())
        t_total_1 = time.perf_counter()
        self.diagnostics["compress_total_time_s"] += float(t_total_1 - t_total_0)

        return compressed_rep

    def _speed_based_representation(self, points: List[Point]) -> List[Tuple]:
        """
        Implements Section 3.1: Speed-Based Representation.
        
        Converts a sequence of points into a sequence of tuples:
        (road_id, direction, offset, speed)
        
        It removes redundant data where speed is constant (within gamma threshold).
        """
        representation: List[Tuple] = []
        if not points:
            return representation

        def lat_lon_dist(p1: Point, p2: Point) -> float:
            # Equirectangular approximation for speed
            lat1 = math.radians(p1.lat)
            lat2 = math.radians(p2.lat)
            dlat = lat2 - lat1
            dlon = math.radians(p2.lon - p1.lon)
            x = dlon * math.cos((lat1 + lat2) / 2.0)
            y = dlat
            return EARTH_RADIUS_M * math.sqrt(x*x + y*y)

        current_road_id = None
        segment_offset = 0.0
        last_stored_speed = -1.0 # Initialize with impossible speed

        for i, p in enumerate(points):
            # Check for road segment change
            if i == 0 or p.road_id != current_road_id:
                current_road_id = p.road_id
                segment_offset = 0.0
                # Start of new road segment: we must record this transition.
                # Speed is implicitly reset or we treat it as 0 initially?
                # Let's record the start point with speed from previous if continuous, 
                # or 0 if we consider it a fresh start. 
                # For simplicity, we record it.
                # We calculate speed to *get here* if possible, or look ahead?
                # TRACE usually suggests V(Tr) tracks speed.
                
                # We'll calculate speed *from* the previous point if it exists and is close in time,
                # but physically this is a new road.
                # Let's just create a start entry.
                speed = 0.0
                # Try to get speed from immediate historical context if available?
                # For now, 0.0 is safe for "start of segment".
                entry = (current_road_id, 1, segment_offset, speed)
                representation.append(entry)
                last_stored_speed = speed
                continue

            # Same road segment
            prev_p = points[i-1]
            dist = lat_lon_dist(prev_p, p)
            segment_offset += dist
            
            time_diff = (p.timestamp - prev_p.timestamp).total_seconds()
            
            if time_diff > 0:
                current_speed = dist / time_diff
            else:
                current_speed = last_stored_speed 

            # Check if speed changed significantly
            if abs(current_speed - last_stored_speed) > self.config.gamma:
                entry = (current_road_id, 1, segment_offset, current_speed)
                representation.append(entry)
                last_stored_speed = current_speed
        
        return representation

    def _referential_compression(self, speed_rep: List[Tuple]) -> Dict[str, List[Any]]:
        """
        Implements Section 3.2: Referential Representation.
        
        Splits the speed-based representation into Edge sequence (E) and Speed sequence (V).
        Compresses each sequence using k-mer matching against the reference set.
        """
        if not speed_rep:
            return {'E': [], 'V': []}
            
        # 1. Extract sequences
        # speed_rep item: (road_id, direction, offset, speed)
        e_seq = [item[0] for item in speed_rep]
        v_seq = [item[3] for item in speed_rep]
        
        # 2. Compress E sequence
        com_e = self._compress_sequence(e_seq, 'E')
        
        # 3. Compress V sequence
        # Apply quantization for speed as per Section 4.2
        # V*(Tr)[i] = round(V(Tr)[i] / (0.5 * eta))
        # We use self.config.epsilon as eta
        eta = self.config.epsilon
        quantized_v_seq = [round(v / (0.5 * eta)) if eta > 0 else int(v) for v in v_seq]
        com_v = self._compress_sequence(quantized_v_seq, 'V')
        
        return {'E': com_e, 'V': com_v}

    def _compress_sequence(self, sequence: List[Any], seq_type: str) -> List[Any]:
        """
        Generic k-mer matching compression for a sequence.
        
        Returns a list of factors:
         - Literal: (value,)
         - Match: (ref_id, start_index, length, mismatch_value)
        """
        compressed = []
        n = len(sequence)
        k = self.config.k
        i = 0
        
        while i < n:
            # Check if enough elements remain for a k-mer
            if i + k > n:
                # Remaining elements are literals
                compressed.append(sequence[i])
                i += 1
                continue
                
            # Form k-mer
            kmer = tuple(sequence[i : i+k])
            kmer_hash = hash((seq_type, kmer))
            
            # Lookup in index
            candidates = self.kmer_index.get(kmer_hash, [])
            
            best_match = None
            max_len = -1
            mismatch_val = None
            
            # Find longest match among candidates
            for ref_id, ref_offset, r_type in candidates:
                if r_type != seq_type:
                    continue
                
                ref = self.references.get(ref_id)
                if not ref:
                    continue
                    
                # Determine which sequence to check in reference
                ref_seq = ref.e_seq if seq_type == 'E' else ref.v_seq # Note: ref.v_seq should be quantized?
                # The reference should store quantized values or we quantize on the fly?
                # Storing processed/quantized sequences in Reference is better.
                # Assuming v_seq in Reference is already compatible (quantized ints)
                
                # Check match length
                match_len = 0
                temp_mismatch = None
                
                # We know first k match (hash collision check needed ideally, but proceed)
                # Verify initial k-mer match to be safe against collisions
                if len(ref_seq) < ref_offset + k:
                    continue
                if tuple(ref_seq[ref_offset : ref_offset + k]) != kmer:
                    continue
                    
                match_len = k
                # Greedy extension
                current_match_idx = i + k
                current_ref_idx = ref_offset + k
                
                while current_match_idx < n and current_ref_idx < len(ref_seq):
                    if sequence[current_match_idx] == ref_seq[current_ref_idx]:
                        match_len += 1
                        current_match_idx += 1
                        current_ref_idx += 1
                    else:
                        temp_mismatch = sequence[current_match_idx]
                        break
                
                if current_match_idx == n and temp_mismatch is None:
                     # End of input sequence, mismatch is implicitly "unknown/None" or next packet
                     temp_mismatch = None # Sentinel
                
                if match_len > max_len:
                    max_len = match_len
                    best_match = (ref_id, ref_offset, match_len, temp_mismatch)
            
            if best_match:
                # Found a match
                compressed.append(best_match)
                # Advance i by match_length plus 1 (since we consumed the mismatch too? 
                # The format is (S, L, M). M is inevitably part of the stream representation.
                # If we emit M, we skip it in input.
                # If M is None (end of stream), we don't increment for M.
                ref_id, start, length, mismatch = best_match
                
                i += length
                if mismatch is not None:
                     # Calculate offset for next iteration
                     i += 1 
            else:
                # No match found, emit literal
                compressed.append(sequence[i])
                
                # NOTE: As per paper, if NO match found, this k-mer becomes a candidate for new reference.
                # The index maintenance is typically done via _manage_references or online.
                # For this implementation, we simply emit the literal.
                i += 1
                
        return compressed

    def _manage_references(self, points: List[Point], speed_rep: List[Tuple], used_refs: set, current_time: float):
        """
        Orchestrates reference maintenance.
        
        1. Updates freshness of used references.
        2. Adds the current trajectory as a new reference (Selection).
        3. Removes old references (Deletion).
        """
        # 1. Update timestamp of used references
        # This implicitly "refreshes" their freshness score to 1.0 (lambda^0)
        for ref_id in used_refs:
            ref = self.references.get(ref_id)
            if ref:
                ref.last_access_time = current_time

        # 2. Add current trajectory as a new reference
        # Extract sequences
        e_seq = [item[0] for item in speed_rep]
        v_raw = [item[3] for item in speed_rep]
        
        # Quantize V sequence for consistent matching
        eta = self.config.epsilon
        v_seq = [round(v / (0.5 * eta)) if eta > 0 else int(v) for v in v_raw]
        
        # Use a simple ID generation
        self.current_ref_id_counter += 1
        new_ref_id = self.current_ref_id_counter
        
        new_ref = Reference(
            ref_id=new_ref_id,
            points=points, # Storing points might be heavy but useful for reconstruction context
            e_seq=e_seq,
            v_seq=v_seq,
            last_access_time=current_time,
            freshness=1.0 # Initial freshness
        )
        self.references[new_ref_id] = new_ref
        
        # Add to index
        self._update_kmer_index([new_ref])
        
        # 3. Check for deletion
        self._reference_deletion(current_time)
        
        # 4. Rewriting
        # self._reference_rewriting(...) 
        # (Skipped for minimal implementation)

    def _reference_deletion(self, current_time: float):
        """
        Implements Algorithm 1: Reference Deletion Algorithm.
        
        Removes references that haven't been used recently to save space.
        """
        t0 = time.perf_counter()
        if not self.references:
            self.diagnostics["reference_delete_time_s"] += 0.0
            return

        decay_lambda = self.config.decay_lambda
        cleanup_threshold_c = 0.5 # 'C' in the paper, using a default constant
        
        # Calculate scores and sum
        # G[i].f = lambda ^ (t_o - G[i].tl)
        # Note: t_o is current_time, G[i].tl is ref.last_access_time
        
        total_freshness = 0.0
        ref_freshness = {} # specific freshness for each ref
        
        refs_to_delete = []
        
        # Current list of references
        all_refs = list(self.references.values())
        
        for ref in all_refs:
            delta_t = current_time - ref.last_access_time
            # Ensure delta_t is non-negative (clock skews?)
            delta_t = max(0.0, delta_t)
            
            # Since delta_t can be large (seconds), lambda should be close to 1
            # or lambda^delta_t vanishes quickly.
            # Paper likely assumes discrete time steps or specific lambda scaling.
            # Assuming lambda is per-second or per-step.
            # If lambda=0.9 and time diff is 100s, score is ~0.
            
            # Using exponential decay
            score = decay_lambda ** delta_t
            ref_freshness[ref.ref_id] = score
            total_freshness += score
            
        avg_freshness = total_freshness / len(all_refs) if all_refs else 0.0
        threshold_value = cleanup_threshold_c * avg_freshness
        
        # Identify outdated references
        # "A reference is outdated ... if its freshness < C * (Fo / |Go|)"
        for ref in all_refs:
            if ref_freshness[ref.ref_id] < threshold_value:
                refs_to_delete.append(ref.ref_id)
                
        # Perform deletion
        for ref_id in refs_to_delete:
            self._delete_reference(ref_id)
        t1 = time.perf_counter()
        self.diagnostics["reference_delete_time_s"] += float(t1 - t0)

    def _delete_reference(self, ref_id: int):
        """Helper to remove a reference and clear its index entries."""
        if ref_id in self.references:
            del self.references[ref_id]
            # Clean up index
            # This is expensive O(IndexSize), but necessary without reverse index.
            # Optimization: Reference object could store which index buckets it's in.
            # For now, we iterate.
            keys_to_remove = []
            for k, candidates in self.kmer_index.items():
                # Filter out the deleted ref_id
                new_candidates = [c for c in candidates if c[0] != ref_id]
                self.kmer_index[k] = new_candidates
                if not new_candidates:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del self.kmer_index[k]

    def _reference_rewriting(self, ref: Reference, index: int, M: int):
        """
        Implements Algorithm 2: Reference Rewriting Algorithm.
        
        Decides whether to replace a specific part of a reference E(Ref)[i] with M.
        
        In this minimal implementation, this is a no-op/placeholder.
        Full implementation requires Factor Matrix tracking which is expensive.
        """
        pass

    def _update_kmer_index(self, references: List[Reference]):
        """
        Helpers to update the k-mer inverted index.
        
        Extracts k-mers from the given references and adds them to the index.
        """
        k = self.config.k
        
        for ref in references:
            # Process Edge Sequence
            n_e = len(ref.e_seq)
            for i in range(n_e - k + 1):
                kmer = tuple(ref.e_seq[i : i+k])
                kmer_hash = hash(('E', kmer))
                # Store (ref_id, offset, type)
                entry = (ref.ref_id, i, 'E')
                self.kmer_index[kmer_hash].append(entry)
                
            # Process Speed Sequence (Quantized)
            n_v = len(ref.v_seq)
            for i in range(n_v - k + 1):
                kmer = tuple(ref.v_seq[i : i+k])
                kmer_hash = hash(('V', kmer))
                entry = (ref.ref_id, i, 'V')
                self.kmer_index[kmer_hash].append(entry)

    def get_diagnostics(self) -> Dict[str, Any]:
        return dict(self.diagnostics)