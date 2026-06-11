"""Shared 1xN trade-off scatter for the appendix calibration figures.

``render_tradeoff`` draws one panel per city plotting an objective against a
feasibility metric. Points are configurations coloured by the swept parameter;
infeasible points are dimmed and the selected operating point is boxed in red.

Each point is a dict with keys ``x``, ``y``, ``color`` (continuous value or
category key), optional ``feasible`` (default True), ``winner`` (default False),
and ``label`` (annotated when ``annotate=True``).
"""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Sequence

from _latex_style import apply_latex_style, recommended_figsize

WINNER_COLOR = "#d62728"


def latest_run(output_root: str, dataset: str, name_re: re.Pattern | None = None) -> str:
    """Return the newest run folder for ``dataset`` ('nyc' or 'sf').

    NYC runs are folders not prefixed ``sf_``; SF runs are those that are.
    ``name_re`` optionally restricts to canonical timestamp folders so
    ablation directories (e.g. ``0519_no_shared_h``) are excluded.
    """
    if not os.path.isdir(output_root):
        raise SystemExit(f"sweep output root missing: {output_root}")
    dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    if name_re is not None:
        dirs = [d for d in dirs if name_re.match(d)]
    dirs = [d for d in dirs if (d.startswith("sf_") if dataset == "sf" else not d.startswith("sf_"))]
    if not dirs:
        raise SystemExit(f"no {dataset} run folder under {output_root}")
    return os.path.join(output_root, max(dirs))


def load_aggregated(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "aggregated.csv"), newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def render_tradeoff(
    panels: Sequence[tuple[str, list[dict]]],
    *,
    x_label: str,
    y_label: str,
    x_log: bool = False,
    y_log: bool = False,
    color_mode: str = "continuous",
    color_label: str | None = None,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    color_log: bool = False,
    discrete_order: list | None = None,
    discrete_labels: dict | None = None,
    thresholds: list[dict] | None = None,
    annotate: bool = False,
    sharey: bool = True,
    legend_title: str | None = None,
    legend_ncol: int | None = None,
):
    apply_latex_style()
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D

    n = len(panels)
    figsize = recommended_figsize(n, 1, full_width=True, panel_aspect=0.92)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=sharey)
    if n == 1:
        axes = [axes]

    if color_mode == "continuous":
        norm = LogNorm(vmin=vmin, vmax=vmax) if color_log else Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        def color_of(p: dict):
            return cmap(norm(p["color"]))
    else:
        order = discrete_order or []
        base = plt.get_cmap("viridis")
        dcolors = {k: base(0.12 + 0.76 * i / max(len(order) - 1, 1)) for i, k in enumerate(order)}

        def color_of(p: dict):
            return dcolors[p["color"]]

    thresholds = thresholds or []

    for ax, (title, points) in zip(axes, panels):
        for thr in thresholds:
            if thr["axis"] == "x":
                ax.axvline(thr["value"], color="#888888", linestyle="--", linewidth=0.9, zorder=2)
            else:
                ax.axhline(thr["value"], color="#888888", linestyle="--", linewidth=0.9, zorder=2)
        for p in points:
            feasible = p.get("feasible", True)
            ax.scatter(
                p["x"], p["y"], color=[color_of(p)], s=46,
                alpha=0.95 if feasible else 0.25,
                edgecolor="white", linewidth=0.5,
                zorder=4 if feasible else 3,
            )
            if annotate and p.get("label"):
                ax.annotate(
                    p["label"], (p["x"], p["y"]), xytext=(5, -2),
                    textcoords="offset points", fontsize=7, color="#333333", zorder=6,
                )
        for p in points:
            if p.get("winner"):
                ax.scatter(
                    p["x"], p["y"], marker="s", s=190, facecolor="none",
                    edgecolor=WINNER_COLOR, linewidth=2.0, zorder=7,
                )
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(y_label)

    for ax in axes:
        for thr in thresholds:
            side = thr.get("infeasible")
            if side is None:
                continue
            if thr["axis"] == "x":
                lo, hi = ax.get_xlim()
                span = (thr["value"], hi) if side == "greater" else (lo, thr["value"])
                ax.axvspan(*span, color=WINNER_COLOR, alpha=0.06, zorder=0)
            else:
                lo, hi = ax.get_ylim()
                span = (thr["value"], hi) if side == "greater" else (lo, thr["value"])
                ax.axhspan(*span, color=WINNER_COLOR, alpha=0.06, zorder=0)

    handles = [
        Line2D([], [], marker="s", color="white", markerfacecolor="none",
               markeredgecolor=WINNER_COLOR, markeredgewidth=2.0, markersize=11,
               linestyle="None", label="selected"),
    ]
    for thr in thresholds:
        if thr.get("label"):
            handles.append(Line2D([], [], color="#888888", linestyle="--", linewidth=0.9, label=thr["label"]))
    if color_mode == "discrete":
        for k in (discrete_order or []):
            handles.append(Line2D(
                [], [], marker="o", color="white", markerfacecolor=dcolors[k],
                markeredgecolor="white", markersize=8, linestyle="None",
                label=(discrete_labels or {}).get(k, str(k)),
            ))
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
        ncol=legend_ncol or min(len(handles), 5), frameon=False, title=legend_title,
    )

    if color_mode == "continuous":
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02, shrink=0.9)
        cbar.set_label(color_label)

    return fig
