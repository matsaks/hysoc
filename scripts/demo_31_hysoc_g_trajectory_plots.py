"""
Demo 31: HYSOC-G Trajectory Plots — Geometric Pipelines

For every trajectory in the input directory, runs three geometric pipelines
and saves a 3-panel figure showing:

  Panel 1 — Raw trajectory (all GPS points, connected line)
  Panel 2 — Oracle-G  (STSS + DP):  move keypoints as dots, stop centroids as stars
  Panel 3 — HYSOC-G   (STEP + SQUISH + DP): same visual encoding

All plots go into a single flat  <output_root>/<timestamp>/plots/  folder.
No per-trajectory subdirectories.

Also produces the same aggregate CSV + agg_summary + contract bundle as Demo 30.
"""

# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pyproj import Transformer
import contextily as ctx

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, current_dir)  # for evaluation_contract

# ---------------------------------------------------------------------------
# src/ imports
# ---------------------------------------------------------------------------

from constants.dp_defaults import DP_DEFAULT_EPSILON_METERS
from constants.segmentation_defaults import (
    STSS_MIN_SAMPLES,
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
)
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY
from core.compression import CompressionStrategy, HYSOCConfig
from core.point import Point
from core.segment import Move, Stop
from core.stream import TrajectoryStream
from engines.dp import DouglasPeuckerCompressor
from engines.stop_compressor import CompressedStop, StopCompressor
from eval.sed import calculate_sed_stats
from hysoc.hysocG import HYSOCCompressor
from oracle.oracleG import OracleG
from evaluation_contract import normalize_pipeline_metrics, write_contract_bundle

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = os.path.join("data", "raw", "NYC_Top_1000_Longest")
DEFAULT_OUTPUT_ROOT = os.path.join("data", "processed", "demo_31_hysoc_g_trajectory_plots")
DEFAULT_BUFFER_CAPACITY = SQUISH_DEFAULT_CAPACITY
DEFAULT_DP_EPSILON_METERS = DP_DEFAULT_EPSILON_METERS

# Visual style constants
COLOR_RAW       = "#546e7a"   # blue-grey for raw points
COLOR_MOVE      = "#1565c0"   # deep blue for move keypoints
COLOR_STOP      = "#e53935"   # vivid red for stop centroids
COLOR_RAW_LINE  = "#90a4ae"   # lighter blue-grey for the connecting line
ALPHA_LINE      = 0.5
ALPHA_PT        = 0.85
MOVE_MARKERSIZE = 30          # scatter s= units
STOP_MARKERSIZE = 120
RAW_MARKERSIZE  = 12
LINE_LW         = 0.8

# Coordinate transformer: WGS-84 lat/lon -> Web Mercator (EPSG:3857)
# always_xy=True means (lon, lat) in -> (x, y) out
_WGS84_TO_WM = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_trajectory(filepath: str, obj_id: str) -> List[Point]:
    stream = TrajectoryStream(
        filepath=filepath,
        col_mapping={"lat": "latitude", "lon": "longitude", "timestamp": "time"},
        default_obj_id=obj_id,
    )
    return list(stream.stream())


def _reconstruct_for_sed(items: list):
    sed_stream, stored_points = [], 0
    for item in items:
        if isinstance(item, CompressedStop):
            p_start = replace(item.centroid, timestamp=item.start_time)
            p_end   = replace(item.centroid, timestamp=item.end_time)
            sed_stream.extend([p_start, p_end])
            stored_points += 1
        elif isinstance(item, Move):
            sed_stream.extend(item.points)
            stored_points += len(item.points)
    return sed_stream, stored_points


def _summarize_sed_and_cr(original, reconstructed, stored_points, latency_us):
    cr = len(original) / max(1, stored_points)
    stats = calculate_sed_stats(original, reconstructed)
    sed_errors = stats.get("sed_errors", [])
    if not sed_errors:
        avg_sed = p95_sed = max_sed = 0.0
    else:
        arr = np.asarray(sed_errors, dtype=float)
        avg_sed  = float(stats["average_sed"])
        p95_sed  = float(np.percentile(arr, 95))
        max_sed  = float(stats["max_sed"])
    return {
        "cr": cr, "stored_points": stored_points,
        "avg_sed_m": avg_sed, "p95_sed_m": p95_sed, "max_sed_m": max_sed,
        "latency_us_per_point": latency_us,
    }


def _metrics_segmented(original, items, latency_us):
    sed_stream, stored = _reconstruct_for_sed(items)
    return _summarize_sed_and_cr(original, sed_stream, stored, latency_us)


def _metrics_hysoc(original, traj_result, latency_us):
    return _summarize_sed_and_cr(
        original, traj_result.keypoints, len(traj_result.keypoints), latency_us
    )


def _to_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _lons_lats(points: List[Point]):
    """Return (lons, lats) lists from a Point sequence."""
    return [p.lon for p in points], [p.lat for p in points]


def _to_mercator(lons, lats):
    """Project WGS-84 lon/lat lists to Web Mercator (x, y) in metres."""
    xs, ys = _WGS84_TO_WM.transform(lons, lats)
    return list(xs), list(ys)


def _draw_raw_panel(ax: plt.Axes, raw_points: List[Point]) -> None:
    """Panel 1 — raw GPS trace with basemap."""
    lons, lats = _lons_lats(raw_points)
    xs, ys = _to_mercator(lons, lats)
    ax.plot(xs, ys, color=COLOR_RAW_LINE, linewidth=LINE_LW,
            alpha=ALPHA_LINE, zorder=1)
    ax.scatter(xs, ys, s=RAW_MARKERSIZE, color=COLOR_RAW,
               alpha=ALPHA_PT, zorder=2, linewidths=0)
    ax.set_title(f"Raw trajectory\n({len(raw_points)} GPS points)", fontsize=11)
    _add_basemap(ax)
    _style_axes(ax)


def _draw_segmented_panel(
    ax: plt.Axes,
    raw_points: List[Point],
    items: list,
    title: str,
) -> None:
    """
    Panel 2 or 3 — compressed segmented view with basemap.

    Draws:
      * Faint raw line as spatial reference
      * Move keypoints as filled circles (COLOR_MOVE), connected by lines
      * Stop centroids as stars (COLOR_STOP), larger
    All coordinates are projected to Web Mercator (EPSG:3857).
    """
    # Faint raw backdrop
    lons_r, lats_r = _lons_lats(raw_points)
    xs_r, ys_r = _to_mercator(lons_r, lats_r)
    ax.plot(xs_r, ys_r, color=COLOR_RAW_LINE, linewidth=LINE_LW * 0.6,
            alpha=0.25, zorder=1)

    move_xs, move_ys = [], []
    stop_xs, stop_ys = [], []

    for item in items:
        if isinstance(item, CompressedStop):
            sx, sy = _to_mercator([item.centroid.lon], [item.centroid.lat])
            stop_xs.extend(sx)
            stop_ys.extend(sy)
        elif isinstance(item, Move):
            lons_m, lats_m = _lons_lats(item.points)
            xs_m, ys_m = _to_mercator(lons_m, lats_m)
            move_xs.extend(xs_m)
            move_ys.extend(ys_m)
            ax.plot(xs_m, ys_m, color=COLOR_MOVE, linewidth=1.1,
                    alpha=0.6, zorder=2)

    if move_xs:
        ax.scatter(move_xs, move_ys, s=MOVE_MARKERSIZE, color=COLOR_MOVE,
                   alpha=ALPHA_PT, zorder=3, linewidths=0)

    if stop_xs:
        ax.scatter(stop_xs, stop_ys, s=STOP_MARKERSIZE, color=COLOR_STOP,
                   marker="*", alpha=ALPHA_PT, zorder=4, linewidths=0)

    ax.set_title(title, fontsize=11)
    _add_basemap(ax)
    _style_axes(ax)

    # Legend (drawn after basemap so it sits on top)
    handles = []
    if move_xs:
        handles.append(mlines.Line2D([], [], color=COLOR_MOVE, marker="o",
                                     markersize=5, linewidth=0.8,
                                     label="Move keypoint"))
    if stop_xs:
        handles.append(mlines.Line2D([], [], color=COLOR_STOP, marker="*",
                                     markersize=9, linewidth=0,
                                     label="Stop centroid"))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.85, borderpad=0.4)


def _add_basemap(ax: plt.Axes) -> None:
    """Add a CartoDB Positron basemap tile layer (silent on failure)."""
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom="auto",
            attribution=False,
        )
    except Exception:
        # No internet / tile server unavailable — continue without tiles
        ax.set_facecolor("#f0f0f0")


def _style_axes(ax: plt.Axes) -> None:
    """Apply a minimal, consistent axis style."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#cccccc")


def _hysoc_to_items(traj_result, stop_compressor: StopCompressor) -> list:
    """
    Convert a TrajectoryResult into a flat list of CompressedStop / Move
    objects (same shape as the Oracle-G item list) for plotting.
    """
    items = []
    for seg in traj_result.segments:
        if seg.kind == "stop":
            # Re-wrap the centroid keypoint as a CompressedStop for uniform handling
            centroid = seg.keypoints[0] if seg.keypoints else None
            if centroid is not None:
                items.append(CompressedStop(
                    centroid=centroid,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                ))
        elif seg.kind == "move":
            items.append(Move(points=seg.keypoints))
    return items


def _save_trajectory_plot(
    out_path: str,
    obj_id: str,
    raw_points: List[Point],
    oracle_items: list,
    hysoc_items: list,
    metrics_oracle: Dict[str, Any],
    metrics_hysoc: Dict[str, Any],
) -> None:
    """Produce and save the 3-panel figure for one trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#ffffff")

    _draw_raw_panel(axes[0], raw_points)

    oracle_n_stops = sum(1 for x in oracle_items if isinstance(x, CompressedStop))
    oracle_n_moves = sum(1 for x in oracle_items if isinstance(x, Move))
    _draw_segmented_panel(
        axes[1], raw_points, oracle_items,
        title=(
            f"Oracle-G  (STSS + DP)\n"
            f"CR={metrics_oracle['cr']:.1f}x   "
            f"SED={metrics_oracle['avg_sed_m']:.1f} m   "
            f"{oracle_n_stops} stops / {oracle_n_moves} moves"
        ),
    )

    hysoc_n_stops = sum(1 for x in hysoc_items if isinstance(x, CompressedStop))
    hysoc_n_moves = sum(1 for x in hysoc_items if isinstance(x, Move))
    _draw_segmented_panel(
        axes[2], raw_points, hysoc_items,
        title=(
            f"HYSOC-G  (STEP + SQUISH + DP)\n"
            f"CR={metrics_hysoc['cr']:.1f}x   "
            f"SED={metrics_hysoc['avg_sed_m']:.1f} m   "
            f"{hysoc_n_stops} stops / {hysoc_n_moves} moves"
        ),
    )

    fig.suptitle(f"Trajectory {obj_id}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo 31: HYSOC-G per-trajectory plots (geometric pipelines)."
    )
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    parser.add_argument("--dp-epsilon-meters", type=float, default=DEFAULT_DP_EPSILON_METERS)
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="If > 0, process only the first N files.",
    )
    args = parser.parse_args()

    input_dir = _to_abs_path(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    csv_files.sort(
        key=lambda name: int(os.path.splitext(name)[0])
        if os.path.splitext(name)[0].isdigit() else 0
    )
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(project_root, args.output_root, timestamp)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Demo 31: HYSOC-G Trajectory Plots")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Plots  : {plots_dir}")
    print(f"  Files  : {len(csv_files)}")
    print(f"  Buffer : {args.buffer_capacity}  eps={args.dp_epsilon_meters} m")
    print()

    # Shared compressor instances
    stop_compressor = StopCompressor()
    dp_compressor   = DouglasPeuckerCompressor(epsilon_meters=args.dp_epsilon_meters)
    oracle_g = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    hysoc_config = HYSOCConfig(
        move_compression_strategy=CompressionStrategy.GEOMETRIC,
        stop_max_eps_meters=STOP_MAX_EPS_METERS,
        stop_min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        squish_buffer_capacity=args.buffer_capacity,
        dp_epsilon_meters=args.dp_epsilon_meters,
    )

    results: List[Dict[str, Any]] = []
    contract_per_file: List[Dict[str, Any]] = []
    online_processing_time_s = 0.0

    for file_idx, fname in enumerate(csv_files, 1):
        obj_id = os.path.splitext(fname)[0]
        path   = os.path.join(input_dir, fname)

        raw_points = _load_trajectory(path, obj_id)
        if len(raw_points) < 2:
            print(f"[{file_idx}/{len(csv_files)}] {obj_id}: skipped (< 2 points)")
            continue

        n_raw = len(raw_points)
        print(f"[{file_idx}/{len(csv_files)}] {obj_id}: {n_raw} pts", end="  ")

        # ---- Oracle-G -------------------------------------------------------
        t0 = time.perf_counter()
        segments_stss = oracle_g.process(raw_points)
        processed_oracle_g: list = []
        for seg in segments_stss:
            if isinstance(seg, Stop):
                processed_oracle_g.append(stop_compressor.compress(seg.points))
            elif isinstance(seg, Move):
                processed_oracle_g.append(Move(points=dp_compressor.compress(seg.points)))
        t1 = time.perf_counter()
        dt_oracle_g = t1 - t0
        online_processing_time_s += dt_oracle_g
        metrics_oracle_g = _metrics_segmented(
            raw_points, processed_oracle_g, (dt_oracle_g * 1e6) / n_raw
        )
        oracle_g_n_stops = sum(1 for x in processed_oracle_g if isinstance(x, CompressedStop))
        oracle_g_n_moves = sum(1 for x in processed_oracle_g if isinstance(x, Move))

        # ---- HYSOC-G --------------------------------------------------------
        compressor_g = HYSOCCompressor(config=hysoc_config)
        t0 = time.perf_counter()
        compressed_g = compressor_g.compress(raw_points)
        t1 = time.perf_counter()
        dt_hysoc_g = t1 - t0
        online_processing_time_s += dt_hysoc_g
        metrics_hysoc_g = _metrics_hysoc(
            raw_points, compressed_g, (dt_hysoc_g * 1e6) / n_raw
        )
        hysoc_g_n_stops = len(compressed_g.stops())
        hysoc_g_n_moves = len(compressed_g.moves())

        print(
            f"OracleG={metrics_oracle_g['cr']:.1f}x/{metrics_oracle_g['avg_sed_m']:.1f}m "
            f"({oracle_g_n_stops}s+{oracle_g_n_moves}m)  "
            f"HYSOC-G={metrics_hysoc_g['cr']:.1f}x/{metrics_hysoc_g['avg_sed_m']:.1f}m "
            f"({hysoc_g_n_stops}s+{hysoc_g_n_moves}m)"
        )

        # ---- Plot -----------------------------------------------------------
        hysoc_items = _hysoc_to_items(compressed_g, stop_compressor)
        plot_path   = os.path.join(plots_dir, f"{obj_id}.png")
        try:
            _save_trajectory_plot(
                out_path=plot_path,
                obj_id=obj_id,
                raw_points=raw_points,
                oracle_items=processed_oracle_g,
                hysoc_items=hysoc_items,
                metrics_oracle=metrics_oracle_g,
                metrics_hysoc=metrics_hysoc_g,
            )
        except Exception as exc:
            print(f"  [warn] Plot failed for {obj_id}: {exc}")

        # ---- Records --------------------------------------------------------
        rec: Dict[str, Any] = {
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            **{f"oracle_g_{k}": v for k, v in metrics_oracle_g.items()},
            "oracle_g_n_stops": oracle_g_n_stops,
            "oracle_g_n_moves": oracle_g_n_moves,
            **{f"hysoc_g_{k}": v for k, v in metrics_hysoc_g.items()},
            "hysoc_g_n_stops": hysoc_g_n_stops,
            "hysoc_g_n_moves": hysoc_g_n_moves,
        }
        results.append(rec)
        contract_per_file.append({
            "obj_id": obj_id,
            "n_raw_points": n_raw,
            "pipelines": {
                "oracle_g": normalize_pipeline_metrics(metrics_oracle_g),
                "hysoc_g":  normalize_pipeline_metrics(metrics_hysoc_g),
            },
        })

    if not results:
        print("No trajectories processed — exiting.")
        return

    # ---- Aggregate CSV ------------------------------------------------------
    csv_path = os.path.join(out_dir, "demo31_evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV           : {csv_path}")

    # ---- agg_summary.json ---------------------------------------------------
    agg_keys = [
        "oracle_g_cr", "hysoc_g_cr",
        "oracle_g_avg_sed_m", "hysoc_g_avg_sed_m",
        "oracle_g_p95_sed_m", "hysoc_g_p95_sed_m",
        "oracle_g_max_sed_m", "hysoc_g_max_sed_m",
        "oracle_g_latency_us_per_point", "hysoc_g_latency_us_per_point",
        "oracle_g_n_stops", "oracle_g_n_moves",
        "hysoc_g_n_stops", "hysoc_g_n_moves",
    ]
    agg: Dict[str, Any] = {"mean": {}, "median": {}}
    for key in agg_keys:
        vals = [float(r[key]) for r in results if key in r]
        agg["mean"][key]   = float(np.mean(vals))   if vals else float("nan")
        agg["median"][key] = float(np.median(vals)) if vals else float("nan")

    agg["timing"] = {
        "provisioning_time_s":       0.0,
        "online_processing_time_s":  online_processing_time_s,
        "end_to_end_time_s":         online_processing_time_s,
        "latency_policy":            "online_primary_with_end_to_end_secondary",
    }

    agg_path = os.path.join(out_dir, "agg_summary.json")
    with open(agg_path, "w", newline="") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved agg_summary   : {agg_path}")

    # ---- Contract bundle ----------------------------------------------------
    contract_paths = write_contract_bundle(
        out_dir,
        script_name="demo_31_hysoc_g_trajectory_plots",
        run_config={
            "input_dir":              input_dir,
            "output_root":            args.output_root,
            "buffer_capacity":        args.buffer_capacity,
            "dp_epsilon_meters":      args.dp_epsilon_meters,
            "max_files":              args.max_files,
            "n_input_files":          len(csv_files),
            "n_processed_files":      len(results),
            "online_processing_time_s": online_processing_time_s,
            "latency_policy":         "online_primary_with_end_to_end_secondary",
        },
        per_file_records=contract_per_file,
        metadata={"notes": "Per-trajectory 3-panel plots (raw/Oracle-G/HYSOC-G)."},
    )
    print(f"Saved contract      : {contract_paths['contract_agg_summary']}")
    print(f"\nPlots directory     : {plots_dir}  ({len(results)} PNG files)")

    print("\n" + "=" * 60)
    print("Demo 31 completed.")
    print(f"Results: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
