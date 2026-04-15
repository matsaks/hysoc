"""Shared evaluation contract helpers for thesis demos.

Phase 1 goal:
- Keep existing demo outputs intact.
- Add a standardized, machine-readable contract bundle with stable keys.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

SCHEMA_VERSION = "hysoc-eval-contract-v1"
PIPELINE_METRIC_KEYS = [
    "cr",
    "stored_points",
    "avg_sed_m",
    "p95_sed_m",
    "max_sed_m",
    "latency_us_per_point",
]


def _to_float(value: Any, default: float = float("nan")) -> float:
    """Convert value to float with NaN-safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_pipeline_metrics(raw: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize arbitrary metric maps into the contract metric keyset."""
    return {key: _to_float(raw.get(key, float("nan"))) for key in PIPELINE_METRIC_KEYS}


def _clean_value(value: Any) -> Any:
    """Convert non-JSON-safe NaN/Inf values to None recursively."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    return value


def aggregate_contract_records(per_file_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate mean/median/p95 per pipeline and metric, ignoring NaNs."""
    pipeline_names: List[str] = []
    for rec in per_file_records:
        for name in rec.get("pipelines", {}).keys():
            if name not in pipeline_names:
                pipeline_names.append(name)

    summary: Dict[str, Any] = {}
    for pipeline in pipeline_names:
        pipeline_summary: Dict[str, Any] = {}
        for metric_key in PIPELINE_METRIC_KEYS:
            vals: List[float] = []
            for rec in per_file_records:
                raw_v = rec.get("pipelines", {}).get(pipeline, {}).get(metric_key, float("nan"))
                v = _to_float(raw_v)
                if not math.isnan(v):
                    vals.append(v)

            if vals:
                arr = np.asarray(vals, dtype=float)
                pipeline_summary[metric_key] = {
                    "mean": float(arr.mean()),
                    "median": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                }
            else:
                pipeline_summary[metric_key] = {"mean": None, "median": None, "p95": None}

        summary[pipeline] = pipeline_summary
    return summary


def write_contract_bundle(
    out_dir: str,
    *,
    script_name: str,
    run_config: Mapping[str, Any],
    per_file_records: List[Dict[str, Any]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Write the standardized contract artifacts for a demo run."""
    os.makedirs(out_dir, exist_ok=True)

    header = {
        "schema_version": SCHEMA_VERSION,
        "script_name": script_name,
        "metric_keys": PIPELINE_METRIC_KEYS,
        "metadata": dict(metadata or {}),
    }
    agg = aggregate_contract_records(per_file_records)

    header_path = os.path.join(out_dir, "contract_header.json")
    run_config_path = os.path.join(out_dir, "run_config.json")
    per_file_path = os.path.join(out_dir, "contract_per_file_metrics.json")
    agg_path = os.path.join(out_dir, "contract_agg_summary.json")

    with open(header_path, "w", newline="") as f:
        json.dump(_clean_value(header), f, indent=2)
    with open(run_config_path, "w", newline="") as f:
        json.dump(_clean_value(dict(run_config)), f, indent=2)
    with open(per_file_path, "w", newline="") as f:
        json.dump(_clean_value(per_file_records), f, indent=2)
    with open(agg_path, "w", newline="") as f:
        json.dump(_clean_value(agg), f, indent=2)

    return {
        "contract_header": header_path,
        "run_config": run_config_path,
        "contract_per_file_metrics": per_file_path,
        "contract_agg_summary": agg_path,
    }
