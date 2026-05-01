# Stop Compression Strategies: Experiment and Results

## 1. Experiment Overview

In the HYSOC pipeline, "Stop" segments (periods where the object is stationary) are compressed into a single representative point. The baseline approach calculates the **Centroid** (arithmetic mean of latitude and longitude) of all points in the stop. 

### The "Drifting Stop" Hypothesis
A potential issue with the Centroid approach is the "drifting stop." When a vehicle pauses at a curved road segment (e.g., waiting at a traffic light on a bend), the GPS fixes scatter along the curve. The geometric centroid of these points often lands inside the arc — physically off the road network.

To investigate whether correcting this behavior improves the Synchronized Euclidean Distance (SED) fidelity, we implemented and evaluated four different single-point stop compression strategies:

1. **Centroid (Baseline):** The arithmetic mean of all points. O(n) complexity.
2. **Snap-to-Nearest:** Calculates the centroid, then snaps to the actual GPS point closest to it. Guarantees the point is a real recorded fix. O(n) complexity.
3. **Medoid:** Finds the actual GPS fix that minimizes the sum of distances to all other points in the stop. O(n²) complexity.
4. **First Point:** Naively takes the first point of the segment. O(1) complexity.

> Note: All methods produce exactly 1 keypoint per stop, meaning the byte cost and Compression Ratio (CR) are identical across all strategies. This isolates the evaluation to just SED accuracy and Processing Latency.

## 2. Implementation Links

Use these files when writing the implementation section of the thesis:

- **Engine Implementation:** [src/engines/stop_compressor.py](file:///c:/Users/mats-/NTNU/hysoc/src/engines/stop_compressor.py)
  *(Contains the logic for all four strategies).*
- **Configuration Defaults:** [src/constants/stop_compression_defaults.py](file:///c:/Users/mats-/NTNU/hysoc/src/constants/stop_compression_defaults.py)
- **Evaluation Script:** [scripts/demo_32_stop_compression_strategies.py](file:///c:/Users/mats-/NTNU/hysoc/scripts/demo_32_stop_compression_strategies.py)
  *(The script that isolated the stop segments and ran the benchmark).*

## 3. Results and Data Links

The experiment was run on the `NYC_Top_1000_Longest` dataset. The evaluation isolated the stop segments to prevent move segment compression (SQUISH+DP) from diluting the error metrics.

### Result Files
- **Aggregated Summary (JSON):** [data/processed/demo_32_stop_compression_strategies/20260430_140403/contract_agg_summary.json](file:///c:/Users/mats-/NTNU/hysoc/data/processed/demo_32_stop_compression_strategies/20260430_140403/contract_agg_summary.json)
- **Per-Trajectory Metrics (CSV):** [data/processed/demo_32_stop_compression_strategies/20260430_140403/demo32_stop_metrics.csv](file:///c:/Users/mats-/NTNU/hysoc/data/processed/demo_32_stop_compression_strategies/20260430_140403/demo32_stop_metrics.csv)
- **Comparison Boxplots (PNG):** [data/processed/demo_32_stop_compression_strategies/20260430_140403/demo32_stop_strategies_comparison.png](file:///c:/Users/mats-/NTNU/hysoc/data/processed/demo_32_stop_compression_strategies/20260430_140403/demo32_stop_strategies_comparison.png)

### Summary of Findings (Mean across all stops)

| Strategy | Avg SED | Max SED | Latency (µs/stop) |
| :--- | :--- | :--- | :--- |
| **First Point** | 14.08 m | 31.55 m | 0.067 µs |
| **Centroid** *(Baseline)* | 4.15 m | 17.60 m | 0.636 µs |
| **Snap-to-Nearest** | 4.15 m | 17.77 m | 0.672 µs |
| **Medoid** | 4.05 m | 18.37 m | 59.32 µs |

## 4. Conclusion for Thesis

When writing the discussion/conclusion for this experiment in your thesis, you can highlight the following points:

1. **The "Drifting Stop" Counter-Intuition:** While forcing the representative point onto the actual road curve (`Snap` or `Medoid`) visually makes sense, it mathematically degrades the worst-case error. The geometric `Centroid` minimizes the maximum distance to all scattered points. By snapping to the edge of the curve, you increase the error distance to the points on the opposite end of the curve, resulting in higher Max SED.
2. **Computational Impracticality of Medoid:** The strict Medoid approach yields a negligible 10cm improvement in Average SED (4.05m vs 4.15m) but incurs a ~100x latency penalty due to its O(n²) complexity. This makes it unsuitable for a high-throughput online streaming system like HYSOC.
3. **Final System Choice:** The baseline `Centroid` strategy is the optimal choice. It provides O(n) performance (<1µs latency), the lowest worst-case error (Max SED), and an average error that is functionally indistinguishable from the more expensive methods.
