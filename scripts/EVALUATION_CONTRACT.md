# HYSOC Evaluation Contract (Phase 1)

This contract standardizes machine-readable outputs across key thesis demos:

- `scripts/demo_16_step_vs_stss_hybrid_ablation.py`
- `scripts/demo_17_hysoc_g_evaluation.py`
- `scripts/demo_21_hysoc_vs_oracles.py`

## Contract Files (written per run directory)

- `contract_header.json`
- `run_config.json`
- `contract_per_file_metrics.json`
- `contract_agg_summary.json`

## Schema

`contract_per_file_metrics.json` entries:

```json
{
  "obj_id": "4494499",
  "n_raw_points": 1245,
  "pipelines": {
    "pipeline_name": {
      "cr": 12.3,
      "stored_points": 101.0,
      "avg_sed_m": 4.2,
      "p95_sed_m": 18.6,
      "max_sed_m": 47.1,
      "latency_us_per_point": 155.0
    }
  }
}
```

## Required Metric Keys

- `cr`
- `stored_points`
- `avg_sed_m`
- `p95_sed_m`
- `max_sed_m`
- `latency_us_per_point`

If a metric is unavailable for a specific demo/pipeline, it is exported as `null` in JSON.

## Aggregation Rules

`contract_agg_summary.json` is produced per pipeline and metric with:

- `mean`
- `median`
- `p95`

Null values are ignored during aggregation.

