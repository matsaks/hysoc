"""
Quick trajectory map visualization with STSS, STEP, and SQUISH processing.

This app loads a raw NYC_100 trajectory, runs:
- STSS (offline) for stop centroids,
- STEP (online) for stop centroids,
- SQUISH on STEP move segments for geometric move compression.

It renders three in-memory map panels in Streamlit:
1) raw trajectory,
2) raw trajectory + STSS stop centroids,
3) raw trajectory + STEP stop centroids (+ STEP+SQUISH move polyline).
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Matplotlib must be headless for Streamlit runs in environments without a GUI.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import geopandas as gpd  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

import contextily as ctx  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from constants.segmentation_defaults import (  # noqa: E402
    STOP_MAX_EPS_METERS,
    STOP_MIN_DURATION_SECONDS,
    STSS_MIN_SAMPLES,
)
from constants.squish_defaults import SQUISH_DEFAULT_CAPACITY  # noqa: E402
from core.point import Point  # noqa: E402
from core.segment import Move, Stop  # noqa: E402
from engines.squish import SquishCompressor  # noqa: E402
from engines.step import STEPSegmenter  # noqa: E402
from oracle.oracleG import OracleG  # noqa: E402


DATA_DIR = REPO_ROOT / "data" / "raw" / "NYC_Top_100_Most_Points"
MAX_PLOTTED_POINTS = 1500
FILENAME_RE = re.compile(r"^[0-9]+$")


def _sanitize_filename(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Filename is empty.")
    if not FILENAME_RE.match(s):
        raise ValueError("Filename must be an integer ID (digits only, no .csv).")
    return s


@st.cache_data(show_spinner=False)
def _list_available_ids() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner=False)
def _load_trajectory_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _detect_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]
    time_candidates = ["time", "timestamp", "datetime", "t", "date"]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            "CSV must contain latitude/longitude columns. "
            f"Found columns: {list(df.columns)}"
        )

    return lat_col, lon_col, time_col


def _downsample_preserving_order(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def _to_point_list(df: pd.DataFrame) -> list[Point]:
    lat_col, lon_col, time_col = _detect_columns(df)
    work = df.copy()
    if time_col is not None:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])
        work = work.sort_values(time_col).reset_index(drop=True)
    else:
        # Fallback for files without explicit timestamps.
        work["__synthetic_time"] = pd.date_range(
            start="2020-01-01 00:00:00", periods=len(work), freq="s"
        )
        time_col = "__synthetic_time"

    points: list[Point] = []
    for _, row in work.iterrows():
        points.append(
            Point(
                lat=float(row[lat_col]),
                lon=float(row[lon_col]),
                timestamp=row[time_col].to_pydatetime()
                if hasattr(row[time_col], "to_pydatetime")
                else datetime.fromisoformat(str(row[time_col])),
                obj_id="streamlit_demo",
            )
        )
    return points


def _extract_centroids(stops: list[Stop]) -> pd.DataFrame:
    rows = []
    for stop in stops:
        if stop.centroid is None:
            continue
        rows.append({"latitude": stop.centroid.lat, "longitude": stop.centroid.lon})
    return pd.DataFrame(rows)


def _run_algorithms(points: list[Point]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stss = OracleG(
        min_samples=STSS_MIN_SAMPLES,
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
        backend="manual",
    )
    stss_segments = stss.process(points)
    stss_stops = [seg for seg in stss_segments if isinstance(seg, Stop)]

    step = STEPSegmenter(
        max_eps=STOP_MAX_EPS_METERS,
        min_duration_seconds=STOP_MIN_DURATION_SECONDS,
    )
    step_segments = []
    for point in points:
        step_segments.extend(step.process_point(point))
    step_segments.extend(step.flush())
    step_stops = [seg for seg in step_segments if isinstance(seg, Stop)]
    step_moves = [seg for seg in step_segments if isinstance(seg, Move)]

    squish = SquishCompressor(capacity=SQUISH_DEFAULT_CAPACITY)
    squish_rows = []
    for move in step_moves:
        compressed_move = squish.compress(move.points)
        for p in compressed_move:
            squish_rows.append({"latitude": p.lat, "longitude": p.lon})

    return (
        _extract_centroids(stss_stops),
        _extract_centroids(step_stops),
        pd.DataFrame(squish_rows),
    )


def _plot_panel(
    ax: plt.Axes,
    raw_df: pd.DataFrame,
    title: str,
    centroid_df: pd.DataFrame | None = None,
    squish_df: pd.DataFrame | None = None,
) -> None:
    lat_col, lon_col, _ = _detect_columns(raw_df)
    raw_plot = _downsample_preserving_order(raw_df, MAX_PLOTTED_POINTS)

    raw_gdf = gpd.GeoDataFrame(
        raw_plot,
        geometry=gpd.points_from_xy(raw_plot[lon_col], raw_plot[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    raw_gdf.plot(ax=ax, color="blue", markersize=6, alpha=0.45)

    if squish_df is not None and not squish_df.empty:
        sq_plot = _downsample_preserving_order(squish_df, MAX_PLOTTED_POINTS)
        if len(sq_plot) >= 2:
            sq_line = LineString(
                [(row["longitude"], row["latitude"]) for _, row in sq_plot.iterrows()]
            )
            sq_gdf = gpd.GeoDataFrame([{"geometry": sq_line}], crs="EPSG:4326").to_crs(
                epsg=3857
            )
            sq_gdf.plot(ax=ax, color="red", linewidth=2.2, alpha=0.9)

    if centroid_df is not None and not centroid_df.empty:
        centroid_gdf = gpd.GeoDataFrame(
            centroid_df,
            geometry=gpd.points_from_xy(centroid_df["longitude"], centroid_df["latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        centroid_gdf.plot(
            ax=ax,
            color="black",
            markersize=110,
            marker="o",
            zorder=6,
        )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title(title)


def _render_three_panel_png(
    raw_df: pd.DataFrame,
    stss_centroids: pd.DataFrame,
    step_centroids: pd.DataFrame,
    squish_points: pd.DataFrame,
    filename_id: str,
) -> BytesIO:
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    _plot_panel(axes[0], raw_df, title=f"Raw trajectory ({filename_id})")
    _plot_panel(
        axes[1],
        raw_df,
        title="Raw + STSS stop centroids",
        centroid_df=stss_centroids,
    )
    _plot_panel(
        axes[2],
        raw_df,
        title="Raw + STEP stop centroids + SQUISH move",
        centroid_df=step_centroids,
        squish_df=squish_points,
    )
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def main() -> None:
    st.set_page_config(page_title="HYSOC STEP+SQUISH Map Viz", layout="wide")
    st.title("HYSOC Quick STEP + SQUISH Map Viz")
    st.caption("Runs STSS, STEP, and SQUISH in memory. No plot files are saved.")

    default_id = "4344893"
    ids = _list_available_ids() or [default_id]

    st.session_state.setdefault("request_plot", False)

    def _request_plot() -> None:
        st.session_state["request_plot"] = True

    st.caption("Filename ID (no .csv)")
    c1, c2 = st.columns([5, 2])
    with c1:
        filename_id = st.selectbox(
            "Filename ID (no .csv)",
            options=ids,
            index=ids.index(default_id) if default_id in ids else 0,
            label_visibility="collapsed",
            key="filename_id_select",
            on_change=_request_plot,
        )
    with c2:
        st.button("Run and plot", type="primary", on_click=_request_plot)

    plot_placeholder = st.empty()
    stats_placeholder = st.empty()

    if st.session_state["request_plot"]:
        st.session_state["request_plot"] = False
        try:
            filename_id = _sanitize_filename(str(filename_id))
            csv_path = DATA_DIR / f"{filename_id}.csv"
            if not csv_path.exists():
                st.error(f"File not found: {csv_path}")
                return

            raw_df = _load_trajectory_csv(csv_path)
            if raw_df.empty:
                st.error("CSV is empty.")
                return

            with st.spinner("Running STSS, STEP, SQUISH and rendering maps..."):
                points = _to_point_list(raw_df)
                stss_centroids, step_centroids, squish_points = _run_algorithms(points)
                png_buf = _render_three_panel_png(
                    raw_df=raw_df,
                    stss_centroids=stss_centroids,
                    step_centroids=step_centroids,
                    squish_points=squish_points,
                    filename_id=filename_id,
                )

            png_bytes = png_buf.getvalue()
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            plot_placeholder.markdown(
                f"""
                <div style="display:flex; justify-content:center;">
                  <img
                    src="data:image/png;base64,{b64}"
                    style="width:1800px; max-width:100%; height:auto;"
                  />
                </div>
                """,
                unsafe_allow_html=True,
            )

            stats_placeholder.info(
                "Run summary - "
                f"raw points: {len(points)} | "
                f"STSS stops: {len(stss_centroids)} | "
                f"STEP stops: {len(step_centroids)} | "
                f"SQUISH move points kept: {len(squish_points)}"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))


if __name__ == "__main__":
    main()
