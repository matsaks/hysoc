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
│       └── utils/              # Shared logic (Geometry, Map-Matching)
│
├── tests/                      # Unit tests (pytest)
├── pyproject.toml              # Dependencies and project config (managed by uv)
└── uv.lock                     # Exact version locking