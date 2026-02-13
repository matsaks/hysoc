# HYSOC: Hybrid Online Semantic Trajectory Compression

**Authors:** Mats Aksnessæther & Jonas Rønning  
**Advisor:** Svein Erik Bratsberg  
**Institution:** NTNU (TDT4900 Master Thesis)

## 📌 Project Overview
This repository contains the implementation of **HYSOC**, a framework for real-time compression of GPS trajectory streams. HYSOC addresses the "Latency-Accuracy Trade-off" by hybridizing behavioral segmentation (STOP/MOVE) with referential compression.

The system is designed to process infinite streams of `(x, y, t)` tuples in real-time, utilizing a modular pipeline:
1.  **Module I:** Behavioral Segmentation (Grid Indexing)
2.  **Module II:** Stop Compression (Semantic Abstraction)
3.  **Module III:** Move Compression (Geometric & Network-based strategies)

## 📂 Project Structure

The project follows a modern Python "Src Layout" to separate source code from experiments and data.

```text
hysoc/
├── benchmarks/                 # Evaluation framework (Chapter 4.6)
│   ├── baselines/              # Competitor algorithms (STEP, TRACE)
│   ├── oracles/                # Offline "Gold Standard" algorithms (STSS, DP)
│   └── metrics.py              # SED, F1-Score, Compression Ratio
│
├── data/                       # Dataset storage (Ignored by Git)
│   ├── raw/                    # Original datasets (WorldTrace, Porto)
│   ├── processed/              # Cleaned streams ready for ingestion
│   └── maps/                   # OSM road networks for map-matching
│
├── notebooks/                  # Jupyter Notebooks for analysis & plotting
│
├── scripts/                    # Executable scripts (e.g., run_experiment.py)
│
├── src/
│   └── hysoc/                  # Main HYSOC Package
│       ├── core/               # Data models (Point, Trajectory, Stream)
│       ├── modules/            # The 3 core architectural components
│       │   ├── segmentation/       # Module I: Grid Index & Stop Detector
│       │   ├── stop_compression/   # Module II: Centroid Abstraction
│       │   └── move_compression/   # Module III: SQUISH (Geom) & TRACE (Net)
│       ├── simulation/         # Streaming Simulation (TrajectorySimulator)
│       └── utils/              # Shared logic (Geometry, Map-Matching)
│
├── tests/                      # Unit tests (pytest)
├── pyproject.toml              # Dependencies and project config (managed by uv)
└── uv.lock                     # Exact version locking

## 🚦 Streaming Simulator
To verify algorithms in real-time without physical devices, HYSOC includes a **Trajectory Simulation** module.

This module reads historical trajectory CSV files and replays them as a live stream of `Point` objects with updated timestamps. 

**Quick Start:**
See `notebooks/demo_simulation.ipynb` for a complete example.

```python
from hysoc.simulation import TrajectorySimulator

# Initialize simulator (obj_id inferred from filename)
sim = TrajectorySimulator("data/raw/subset_50/4325685.csv", interval=1.0)

# Simulate device stream
for point in sim.stream():
    print(f"Received: {point}")
    # Feed 'point' into HYSOC pipeline...
```

## 🛑 Stop Compression Demo
To visualize the Stop Compression module (Module II) in action, you can run the provided demo script. This script performs the following steps:
1.  **Loads** a real GPS trajectory from `data/raw/subset_50/`.
2.  **Segments** the trajectory into Stops and Moves using `STSSOracleSklearn`.
3.  **Compresses** the Stop segments into single centroids with start/end times.
4.  **Visualizes** the result with a side-by-side map comparison (Raw vs. Compressed).

**Run the demo:**
```bash
uv run scripts/demo_stop_compression.py
```

**Output:**
The script saves the results to `data/processed/` with filenames based on the input trajectory ID:
-   `compressed_output_[id].csv`: CSV file containing the compressed stream (Stop Centroids + Move Points).
-   `stop_compression_demo_[id].png`: Visualization showing raw GPS points vs. the compressed trajectory.
