# HYSOC — Hybrid Online Semantic Compression System

HYSOC is a fully online framework for semantic trajectory compression. A streaming
STOP/MOVE segmenter (STEP) feeds two complementary strategies:

- **HYSOC-G** — geometric compression via hybrid SQUISH + Douglas–Peucker.
- **HYSOC-N** — network-semantic compression via TRACE (k-mer referential matching
  over pre-annotated road IDs).

Both are benchmarked against offline oracles (Baseline-G: STSS + Douglas–Peucker;
Baseline-N: STSS + STC) on the WorldTrace dataset, trading off compression ratio,
information preservation, and per-element latency. This package contains the code,
datasets, and generated results behind the thesis.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A LaTeX installation (e.g. MiKTeX or TeX Live) — only the thesis figure scripts need
  it, as they render text through matplotlib's LaTeX backend. Sweeps and experiments
  run without it.

## Setup

```bash
uv sync
```

This builds the pinned environment from `pyproject.toml` and `uv.lock`. Run everything
through `uv run` so that environment is used; `src/` is placed on the import path
automatically, so imports such as `engines.trace` resolve without extra configuration.

## Repository layout

```
.
├── README.md                this file
├── pyproject.toml           project metadata, dependencies, authors
├── uv.lock                  pinned dependency versions
├── data/
│   └── raw/                 GPS trajectory datasets (one CSV per trajectory)
│       ├── NYC_Calibration_100/
│       ├── NYC_Evaluation_1000/
│       ├── SanFranCentre_Calibration_100/
│       └── SanFranCentre_Evaluation_1000/
├── src/
│   ├── main.py              command-line entry point
│   ├── engines/             compression algorithms (STEP, STSS, SQUISH, DP, TRACE, STC)
│   ├── hysoc/               HYSOC-G / HYSOC-N orchestrators
│   ├── oracle/              offline baselines (Oracle-G, Oracle-N)
│   ├── eval/                metrics: CR, SED, Stop F1, road Jaccard, latency
│   ├── sweeps/              parameter-selection sweep drivers
│   ├── experiments/         end-to-end evaluation drivers
│   ├── constants/           calibrated parameters and per-city operating points
│   ├── core/                trajectory primitives and shared types
│   └── io/                  dataset loading and streaming
├── scripts/
│   ├── thesis_figures/      one generator per thesis figure (NN_*.py)
│   └── data/                dataset-preparation utilities
└── results/
    ├── sweeps/              calibration-sweep outputs (CSV + run_config.json)
    ├── experiments/         end-to-end run outputs (CSV + comparison PNGs)
    └── figures/             thesis figure PNGs and compute caches
```

The 1,100 longest WorldTrace trajectories inside each of the New York City and San
Francisco bounding boxes were split into a 100-trajectory calibration subset (used only
for parameter selection) and a 1,000-trajectory evaluation subset (used for all reported
benchmarks).

## Reproducing the results

All paths resolve relative to this folder, and the drivers write under `results/`. The
shipped `results/` already holds the runs and figures behind the reported numbers, so
the figure scripts reproduce them without re-running the sweeps and experiments first.

### 1. Calibration sweeps

```bash
# Module I/II/III HYSOC-G sweeps under one shared timestamp:
uv run python src/sweeps/run_all.py --dataset nyc
uv run python src/sweeps/run_all.py --dataset sf_centre

# Auxiliary sweeps (standalone):
uv run python src/sweeps/module4_hysoc_n_trace.py --dataset nyc
uv run python src/sweeps/oracle_g_dp_epsilon.py --dataset nyc
```

`--dataset` accepts `nyc` or `sf_centre`. Output: `results/sweeps/<sweep>/<MMDD_HHMM>/`.

### 2. End-to-end evaluation

```bash
# HYSOC-G vs Baseline-G vs Plain DP:
uv run python src/main.py experiment hysoc_g --operating-point nyc
uv run python src/main.py experiment hysoc_g --operating-point sf_centre

# HYSOC-N vs Baseline-N vs Plain TRACE:
uv run python src/main.py experiment hysoc_n --operating-point nyc
uv run python src/main.py experiment hysoc_n --operating-point sf_centre
```

`--operating-point` accepts `nyc` or `sf_centre`; add `--max-files N` for a quick smoke
test. Output: `results/experiments/<exp>/<MMDD_HHMM>_op-<point>/`.

Compress a single trajectory:

```bash
uv run python src/main.py compress --input data/raw/NYC_Evaluation_1000/<file>.csv --mode hysoc_g
```

### 3. Figures

`scripts/thesis_figures/` holds one generator per thesis figure; run any of them the same
way, and each writes its PNG to `results/figures/<name>/<MMDD_HHMM>/`.

## Evaluation metrics

| Metric            | Definition                                                | Applies to          |
| ----------------- | --------------------------------------------------------- | ------------------- |
| Compression ratio | `original_bytes / encoded_bytes` (24 bytes per raw point) | all systems         |
| Stop F1           | temporal-IoU matching vs the STSS reference, IoU ≥ 0.5    | all systems         |
| SED               | synchronised Euclidean distance                           | HYSOC-G, Baseline-G |
| Road Jaccard      | road-segment set overlap                                  | HYSOC-N, Baseline-N |
| Latency           | median per-element processing time (microseconds)         | all systems         |
