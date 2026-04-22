# HYSOC: Hybrid Online Semantic Trajectory Compression

**Authors:** Mats Aksnessæther & Jonas Rønning  
**Advisor:** Svein Erik Bratsberg  
**Institution:** NTNU (TDT4900 Master Thesis)

## Project Overview

This repository contains the implementation of **HYSOC**, a framework for real-time compression of GPS trajectory streams. HYSOC addresses the latency–accuracy trade-off by hybridising behavioural segmentation (STOP/MOVE) with dual compression strategies.

The system processes infinite streams of `(x, y, t)` tuples in real-time via a three-module pipeline:

1. **Module I — Streaming Segmenter**: Real-time STOP/MOVE detection using grid indexing.
2. **Module II — Stop Compressor**: Semantic abstraction to a single centroid + timestamps.
3. **Module III — Move Compressor**: Two strategies — **HYSOC-G** (geometric, SQUISH) and **HYSOC-N** (network-semantic, TRACE).

## Project Structure

```text
hysoc/
├── thesis/                     # Overleaf (LaTeX thesis) submodule
├── src/                        # Core Python package
│   ├── main.py
│   ├── hysoc/
│   │   ├── hysocG.py           # HYSOC-G (geometric, SQUISH)
│   │   └── hysocN.py           # HYSOC-N (network-semantic, TRACE)
│   ├── engines/                # Online/offline compression engines
│   ├── oracle/                 # Oracle baselines (STSS + DP / STC)
│   ├── eval/                   # Evaluation metrics
│   │   ├── compression.py      # Compression ratio
│   │   ├── sed.py              # Synchronized Euclidean Distance
│   │   └── segmentation.py     # Stop F1, temporal IoU, road-segment Jaccard
│   ├── core/                   # Shared primitives
│   │   ├── compression.py      # TrajectoryResult, SegmentResult, BYTES_PER_POINT
│   │   └── ...
│   ├── io/                     # Data loading / serialisation
│   └── constants/
├── scripts/                    # Experiment and demo drivers
├── tests/                      # pytest suite
├── data/                       # Raw/processed trajectory datasets (git-ignored)
├── papers/                     # PDFs and per-paper reading summaries
├── webapps/                    # Visual tooling
├── pyproject.toml
└── uv.lock
```

## Evaluation Strategy

HYSOC uses strategy-appropriate metrics because HYSOC-G and HYSOC-N compress fundamentally different representations.

| Metric | HYSOC-G | HYSOC-N | Oracle-G | Oracle-N |
|---|---|---|---|---|
| Compression Ratio (bytes) | ✓ | ✓ | ✓ | ✓ |
| Stop F₁ (temporal IoU ≥ 0.5) | ✓ | ✓ | ✓ | ✓ |
| SED | ✓ | — | ✓ | — |
| Road-segment Jaccard | — | ✓ | — | ✓ |

**Compression Ratio** is byte-based (`original_bytes / encoded_bytes`, `BYTES_PER_POINT = 24`) so TRACE encoding cost is reflected fairly rather than as a point count.

**SED** is not applicable to HYSOC-N: TRACE is lossless with respect to the map-matched representation, so SED against raw GPS would measure map-matching error rather than compression error.

**Road-segment Jaccard** compares the sets of `road_id` values in each result's keypoints. Returns `nan` when no road IDs are present so that datasets without ground-truth road IDs do not silently produce a perfect score.

## Core Types

All strategies produce a `TrajectoryResult`; all eval code operates on this single type (`src/core/compression.py`).

- **`BYTES_PER_POINT = 24`** — lat (float64=8) + lon (float64=8) + timestamp (int64=8).
- **`SegmentResult`** (frozen dataclass) — `kind: Literal["stop","move"]`, `start_time`, `end_time`, `keypoints: list[Point]`, `encoded_bytes: int`. For point-list strategies `encoded_bytes = len(keypoints) * BYTES_PER_POINT`; for TRACE it is the actual encoding size.
- **`TrajectoryResult`** — `object_id`, `original_points`, `segments: list[SegmentResult]`, `strategy`. Properties: `keypoints` (flat reconstruction), `original_bytes`, `encoded_bytes`, `compression_ratio`. Methods: `stops()`, `moves()`.

## Submodule: Thesis (Overleaf)

The `thesis/` directory is a Git submodule linked to Overleaf. Use `/fetch_overleaf` and `/push_overleaf` slash commands in Claude Code to sync changes.

### First-time setup (after cloning)

```bash
git submodule update --init thesis
```

### Manual pull (Overleaf → local)

```bash
git submodule update --remote thesis
git add thesis
git commit -m "update thesis submodule pointer"
```

### Manual push (local → Overleaf)

```bash
git -C thesis add .
git -C thesis commit -m "Your message"
git -C thesis push origin HEAD:master
git add thesis
git commit -m "update thesis submodule pointer"
```
