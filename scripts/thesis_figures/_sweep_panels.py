"""Shared matplotlib helper for sweep figures.

``render_sweep_panels`` draws a 1xN row of metric panels for a parameter sweep,
plotting each metric's median against the swept parameter and optionally grouping
into multiple curves by a categorical column. Purely presentational.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from _latex_style import apply_latex_style


@dataclass(frozen=True)
class PanelSpec:
    """One metric panel within a sweep figure."""

    column: str
    ylabel: str
    h_ref: float | None = None
    h_ref_label: str | None = None
    log_y: bool = False
    y_shade_above: float | None = None
    y_shade_label: str | None = None
    y_shade_color: str = "#d62728"


@dataclass(frozen=True)
class BreakMarker:
    """An off-axis control point drawn to the left of a log x-axis."""

    x_label: str
    values: dict[str, float]
    color: str = "#d62728"


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def _filter_finite(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def _group_rows(
    rows: list[dict], group_col: str | None
) -> list[tuple[str | None, list[dict]]]:
    if group_col is None:
        return [(None, list(rows))]
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        key = str(r[group_col])
        grouped.setdefault(key, []).append(r)

    def sort_key(item: tuple[str, list[dict]]) -> float:
        try:
            return float(item[0])
        except ValueError:
            return float("inf")

    return [(k, v) for k, v in sorted(grouped.items(), key=sort_key)]


def render_sweep_panels(
    rows: list[dict],
    x_col: str,
    panels: Sequence[PanelSpec],
    *,
    x_label: str,
    group_col: str | None = None,
    group_label_fmt: str = "{value}",
    log_x: bool = False,
    winner: dict[str, float] | None = None,
    infeasible_col: str | None = None,
    break_marker: BreakMarker | None = None,
    figsize: tuple[float, float] = (15.0, 3.6),
    legend_title: str | None = None,
    annotate_points: bool = False,
    annotate_fmt: str = "{value:g}",
    logger: logging.Logger | None = None,
):
    """Render an N-panel sweep figure (one :class:`PanelSpec` per panel).

    ``group_col`` produces multiple curves; ``winner`` highlights the selected
    operating point; ``infeasible_col`` dims flagged rows while keeping the line
    continuous; ``break_marker`` adds an off-axis control point (e.g. beta=0).
    """

    apply_latex_style()

    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, ScalarFormatter

    if logger is None:
        logger = logging.getLogger(__name__)

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    groups = _group_rows(rows, group_col)
    cmap = plt.get_cmap("viridis")
    n_groups = max(len(groups), 1)
    group_colors = {
        key: cmap(0.15 + 0.75 * i / max(n_groups - 1, 1))
        for i, (key, _) in enumerate(groups)
    }
    if n_groups == 1:
        group_colors[groups[0][0]] = "#1f77b4"

    for panel_idx, panel in enumerate(panels):
        ax = axes[panel_idx]

        for key, gr_rows in groups:
            gr_rows_sorted = sorted(gr_rows, key=lambda r: _to_float(r[x_col]))
            xs = [_to_float(r[x_col]) for r in gr_rows_sorted]
            ys = [_to_float(r[panel.column]) for r in gr_rows_sorted]
            xs_finite, ys_finite = _filter_finite(xs, ys)
            if not xs_finite:
                logger.warning(
                    "panel %s group %s has no finite data for column %s",
                    panel.column,
                    key,
                    panel.column,
                )
                continue

            color = group_colors[key]
            label = (
                group_label_fmt.format(value=key) if group_col is not None else None
            )
            ax.plot(
                xs_finite,
                ys_finite,
                "-",
                color=color,
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
                label=label,
            )

            for r, x, y in zip(gr_rows_sorted, xs, ys):
                if math.isnan(x) or math.isnan(y):
                    continue
                infeasible = False
                if infeasible_col is not None:
                    raw = r.get(infeasible_col, "")
                    if isinstance(raw, str):
                        raw = raw.strip().lower()
                        infeasible = raw in {"", "0", "0.0", "false", "no", "nan"}
                    else:
                        infeasible = not bool(raw)
                ax.plot(
                    x,
                    y,
                    "o",
                    color=color,
                    markersize=4.5,
                    alpha=0.35 if infeasible else 0.95,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=3,
                )
                if annotate_points:
                    ax.annotate(
                        annotate_fmt.format(value=x),
                        xy=(x, y),
                        xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color="#444",
                        alpha=0.7,
                        zorder=4,
                    )

        if panel.h_ref is not None:
            ax.axhline(
                panel.h_ref,
                color="#888888",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
                label=panel.h_ref_label,
            )

        if panel.y_shade_above is not None:
            ax.autoscale_view()
            y_lo, y_hi = ax.get_ylim()
            if y_hi > panel.y_shade_above:
                ax.axhspan(
                    panel.y_shade_above,
                    y_hi,
                    color=panel.y_shade_color,
                    alpha=0.08,
                    zorder=0,
                    label=panel.y_shade_label,
                )
                ax.set_ylim(y_lo, y_hi)

        if winner is not None:
            win_x = _to_float(winner.get(x_col))
            win_y_rows = [
                r
                for r in rows
                if math.isclose(_to_float(r[x_col]), win_x, rel_tol=1e-9, abs_tol=1e-9)
                and (
                    group_col is None
                    or str(r.get(group_col)) == str(winner.get(group_col))
                )
            ]
            if win_y_rows:
                win_y = _to_float(win_y_rows[0][panel.column])
                if not math.isnan(win_y):
                    ax.plot(
                        win_x,
                        win_y,
                        "s",
                        color="#d62728",
                        markersize=10,
                        markeredgecolor="white",
                        markeredgewidth=1.2,
                        zorder=5,
                        label="selected" if panel_idx == 0 else None,
                    )

        if break_marker is not None and panel.column in break_marker.values:
            bm_y = break_marker.values[panel.column]
            if not (math.isnan(bm_y) or math.isinf(bm_y)):
                cur_xmin, cur_xmax = ax.get_xlim()
                if log_x:
                    finite_xs = [
                        _to_float(r[x_col]) for r in rows
                        if not math.isnan(_to_float(r[x_col]))
                        and _to_float(r[x_col]) > 0
                    ]
                    data_min = min(finite_xs) if finite_xs else 1.0
                    break_x = data_min / 3.0
                else:
                    finite_xs = [
                        _to_float(r[x_col]) for r in rows
                        if not math.isnan(_to_float(r[x_col]))
                    ]
                    data_min = min(finite_xs) if finite_xs else 0.0
                    data_max = max(finite_xs) if finite_xs else 1.0
                    break_x = data_min - 0.08 * (data_max - data_min)
                ax.plot(
                    break_x,
                    bm_y,
                    "D",
                    color=break_marker.color,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=4,
                    label=break_marker.x_label if panel_idx == 0 else None,
                )
                ax.set_xlim(break_x * (0.6 if log_x else 1.0), cur_xmax)

        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if panel.log_y:
            ax.set_yscale("log")

        ax.set_xlabel(x_label)
        ax.set_ylabel(panel.ylabel)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.15)

    legend_handles_labels: dict[str, Any] = {}
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl and lbl not in legend_handles_labels:
                legend_handles_labels[lbl] = h

    if legend_handles_labels:
        fig.legend(
            legend_handles_labels.values(),
            legend_handles_labels.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_handles_labels), 6),
            frameon=False,
            title=legend_title,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
