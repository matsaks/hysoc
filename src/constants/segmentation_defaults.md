# Segmentation Defaults: Experiment Summary

This note documents how the current values in `segmentation_defaults.py` were selected from empirical sweeps on the NYC subset.

## Scripts Used

- `scripts/demo_27_step_stss_param_sweep.py`
  - Runs STEP and STSS on all trajectories for a parameter grid.
  - Writes per-trajectory results to `data/processed/param_sweep_step_stss.csv`.
- `scripts/demo_28_aggregate_param_sweep.py`
  - Aggregates per-trajectory rows into one row per configuration.
  - Used in `--slim` mode for direct comparison:
    `data/processed/param_sweep_step_stss_aggregated_slim.csv`.

## Dataset and Sweep Setup

- Dataset: `data/raw/NYC_Top_100_Most_Points/`
- Number of trajectories: 100
- Total points processed: 163,684
- Sweep grid:
  - `eps_m in {10, 15, 20, 30, 50}`
  - `t_s in {10, 15, 30, 60}`
- Shared thresholds for comparability:
  - STEP and STSS use same `eps_m` and `t_s`.
- STSS MinPts derivation:
  - `min_samples = max(5, round(t_s * 0.5))` (1 Hz assumption)

## Metrics Reported

Per trajectory (from `demo_27`):
- STEP/STSS stop and move counts
- F1, precision, recall
- Mean matched temporal IoU
- Runtime

Aggregated per config (from `demo_28`):
- Means/medians/std across trajectories
- Heuristic flag: `passes_plan_heuristic`
  - `median step stops >= 2`
  - `median step moves >= 2`
  - `median F1 > 0.6`

## Key Results (from aggregated slim CSV)

Top configurations with `f1_median > 0.7`:

| eps_m | t_s | step_n_stops_mean | step_n_moves_mean | f1_median | matched_iou_mean_median |
|---:|---:|---:|---:|---:|---:|
| 10 | 30 | 3.60 | 4.17 | 0.800 | 0.934 |
| 15 | 30 | 3.84 | 4.30 | 0.760 | 0.929 |
| 20 | 60 | 1.83 | 2.46 | 0.733 | 0.951 |
| 30 | 30 | 4.09 | 4.46 | 0.727 | 0.908 |

Interpretation:
- `10,30` gives the highest agreement (best F1).
- `30,30` gives the richest segmentation, but lower F1 than `10,30`/`15,30`.
- `20,60` is accurate but produces fewer stop/move segments.
- `15,30` is the selected compromise between quality and segment richness.

## Selected Defaults

Current defaults in `segmentation_defaults.py`:

- `STOP_MAX_EPS_METERS = 15.0`
- `STSS_MAX_EPS_METERS = 15.0`
- `STOP_MIN_DURATION_SECONDS = 30.0`
- `STSS_MIN_DURATION_SECONDS = 30.0`
- `STSS_MIN_SAMPLES = 15`

Rationale:
- Maintains high median agreement (`f1_median ~= 0.76`, IoU median `~= 0.93`).
- Produces slightly more stop/move segments than `10,30`, which benefits downstream HYSOC behavior that depends on stop boundaries for move-buffer lifecycle events.

## Reproduction Commands

Run sweep:

```bash
uv run python scripts/demo_27_step_stss_param_sweep.py
```

Aggregate (full):

```bash
uv run python scripts/demo_28_aggregate_param_sweep.py --out data/processed/param_sweep_step_stss_aggregated.csv
```

Aggregate (slim comparison sheet):

```bash
uv run python scripts/demo_28_aggregate_param_sweep.py --slim --out data/processed/param_sweep_step_stss_aggregated_slim.csv
```
