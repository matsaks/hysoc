"""
Quick trajectory map visualization (raw GPS only).

This app intentionally does *not* run HMM map-matching and does *not* save plots to disk.
It loads `data/raw/NYC_100/<filename>.csv`, plots the trajectory on top of a basemap,
and renders the result directly in Streamlit from an in-memory PNG.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import re

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
    # webapps/quick_trajectory_map_viz/app.py -> repo root is 2 levels up.
    # parents[0] = quick_trajectory_map_viz
    # parents[1] = webapps
    # parents[2] = hysoc (repo root)
    return Path(__file__).resolve().parents[2]


DATA_DIR = _repo_root() / "data" / "raw" / "NYC_100"

MAX_PLOTTED_POINTS = 1500


FILENAME_RE = re.compile(r"^[0-9]+$")


def _sanitize_filename(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Filename is empty.")

    # Basic safety: dataset filenames are numeric IDs in this repo.
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
    # Raw CSVs in this repo typically use: time, latitude, longitude
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

    # Evenly sample indices to keep trajectory direction roughly intact.
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def _render_map_png(df: pd.DataFrame, filename_id: str) -> BytesIO:
    lat_col, lon_col, time_col = _detect_columns(df)

    df_plot = _downsample_preserving_order(df, max_points=MAX_PLOTTED_POINTS)

    if time_col is not None:
        # Sorting by time yields a more correct polyline ordering for most datasets.
        df_plot = df_plot.copy()
        df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
        df_plot = df_plot.sort_values(time_col).reset_index(drop=True)

    # Build point geometries in EPSG:4326 (lon/lat), then project to 3857 for basemap.
    if len(df_plot) < 1:
        raise ValueError("Trajectory has no points to plot.")

    points = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot[lon_col], df_plot[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Similar styling to `demo_02_offline_stss_and_squish.py` plot 1:
    # larger canvas, but keep the points readable (blue dots with transparency).
    fig, ax = plt.subplots(figsize=(12, 8))
    points.plot(ax=ax, color="blue", markersize=5, alpha=0.5)

    # Add web tiles basemap (internet required).
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title(f"Raw trajectory (NYC_100/{filename_id}.csv)")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def main() -> None:
    st.set_page_config(page_title="HYSOC Quick Map Viz", layout="centered")
    st.title("HYSOC Quick Trajectory Map Viz")
    st.caption("Raw GPS plotting only. No plot files saved.")

    default_id = "4344893"

    ids = _list_available_ids()
    if not ids:
        ids = [default_id]

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
        st.button("Plot trajectory", type="primary", on_click=_request_plot)

    # Render into a single placeholder so repeated submissions don't stack images.
    plot_placeholder = st.empty()

    if st.session_state["request_plot"]:
        st.session_state["request_plot"] = False
        try:
            filename_id = _sanitize_filename(str(filename_id))
            csv_path = DATA_DIR / f"{filename_id}.csv"

            if not csv_path.exists():
                st.error(f"File not found: {csv_path}")
                return

            df = _load_trajectory_csv(csv_path)
            if df.empty:
                st.error("CSV is empty.")
                return

            with st.spinner("Rendering map (basemap tiles may take a moment)..."):
                png_buf = _render_map_png(df, filename_id=filename_id)

            # `st.image` tends to left-align within its container.
            # Render via HTML to guarantee centering while keeping a large size.
            png_bytes = png_buf.getvalue()
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            caption = f"NYC_100/{filename_id}.csv"
            plot_placeholder.markdown(
                f"""
                <div style="display:flex; justify-content:center;">
                  <img
                    src="data:image/png;base64,{b64}"
                    style="width:1100px; max-width:100%; height:auto;"
                  />
                </div>
                <div style="text-align:center; font-size: 0.9em; opacity: 0.9; margin-top: 4px;">
                  {caption}
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:  # noqa: BLE001
            st.error(str(e))


if __name__ == "__main__":
    main()

