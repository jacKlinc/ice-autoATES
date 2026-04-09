from __future__ import annotations

import math
from datetime import datetime, date, time

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from ates.advisory import wet_avalanche_risk
from ates.areas import ATES_HEX, ATES_LABELS, load_areas
from ates.gpx import parse_gpx
from ates.sample import find_area_for_points, sample_ates

st.set_page_config(page_title="Route Analysis — AutoATES", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATES_FOLIUM_COLOUR = {
    1: "green",
    2: "blue",
    3: "black",
    4: "red",
    -1: "gray",
}


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _cumulative_distance(points: list[dict]) -> list[float]:
    dist = [0.0]
    for i in range(1, len(points)):
        d = _haversine_m(
            points[i - 1]["lat"], points[i - 1]["lon"],
            points[i]["lat"], points[i]["lon"],
        )
        dist.append(dist[-1] + d)
    return dist


def _build_route_map(area: dict, points: list[dict], classes: list[int]) -> folium.Map:
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # Draw route segments coloured by ATES class
    for i in range(1, len(points)):
        cls = classes[i]
        colour = _ATES_FOLIUM_COLOUR.get(cls, "gray")
        folium.PolyLine(
            locations=[
                [points[i - 1]["lat"], points[i - 1]["lon"]],
                [points[i]["lat"], points[i]["lon"]],
            ],
            color=colour,
            weight=4,
            opacity=0.85,
            tooltip=f"ATES {cls}: {ATES_LABELS.get(cls, 'Unknown')}",
        ).add_to(m)

    # Start / end markers
    folium.Marker(
        location=[points[0]["lat"], points[0]["lon"]],
        tooltip="Start",
        icon=folium.Icon(color="white", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=[points[-1]["lat"], points[-1]["lon"]],
        tooltip="End",
        icon=folium.Icon(color="white", icon="flag", prefix="fa"),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _elevation_profile(points: list[dict], classes: list[int]) -> alt.Chart | None:
    elevations = [p.get("ele") for p in points]
    if all(e is None for e in elevations):
        return None

    dists = _cumulative_distance(points)
    df = pd.DataFrame(
        {
            "distance_km": [d / 1000 for d in dists],
            "elevation_m": [e if e is not None else float("nan") for e in elevations],
            "ates_class": [str(c) if c >= 0 else "Unknown" for c in classes],
            "ates_label": [ATES_LABELS.get(c, "Unknown") for c in classes],
        }
    )

    colour_scale = alt.Scale(
        domain=["1", "2", "3", "4", "Unknown"],
        range=[ATES_HEX[1], ATES_HEX[2], ATES_HEX[3], ATES_HEX[4], "#aaaaaa"],
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X("distance_km:Q", title="Distance (km)"),
            y=alt.Y("elevation_m:Q", title="Elevation (m)", scale=alt.Scale(zero=False)),
            color=alt.Color("ates_class:N", scale=colour_scale, title="ATES class"),
            tooltip=[
                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("elevation_m:Q", title="Elevation (m)", format=".0f"),
                alt.Tooltip("ates_label:N", title="ATES"),
            ],
        )
        .properties(height=200, title="Elevation profile (coloured by ATES)")
    )
    return chart


def _advisory_banner(advisory: dict):
    level = advisory["level"]
    reason = advisory["reason"]
    if level == "elevated":
        st.error(f"Wet avalanche risk: **Elevated** — {reason}")
    elif level == "moderate":
        st.warning(f"Wet avalanche risk: **Moderate** — {reason}")
    else:
        st.info(f"Wet avalanche risk: **Low** — {reason}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Route Analysis")

areas = load_areas()

uploaded = st.sidebar.file_uploader("Upload GPX track", type=["gpx"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Trip date & time")
trip_date = st.sidebar.date_input("Date", value=date.today())
trip_time = st.sidebar.time_input("Start time (local)", value=time(7, 0))

st.sidebar.markdown("---")
st.sidebar.markdown("### ATES legend")
for cls in range(1, 5):
    hex_col = ATES_HEX[cls]
    label = ATES_LABELS[cls]
    st.sidebar.markdown(
        f"<span style='color:{hex_col}; font-size:1.2em'>■</span> **{cls}** — {label}",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Route Analysis")

if uploaded is None:
    st.info("Upload a GPX file in the sidebar to analyse your route.")
    st.stop()

# Parse GPX
raw = uploaded.read()
try:
    points = parse_gpx(raw)
except Exception as e:
    st.error(f"Could not parse GPX file: {e}")
    st.stop()

if not points:
    st.error("No track points found in the GPX file.")
    st.stop()

# Wet avalanche advisory
trip_dt = datetime.combine(trip_date, trip_time)
advisory = wet_avalanche_risk(trip_dt)
_advisory_banner(advisory)

# Find matching area
area = find_area_for_points(areas, points)
if area is None:
    area_names = [a["name"] for a in areas]
    selected_name = st.selectbox(
        "Route is outside known areas — select manually:",
        area_names,
    )
    area = next(a for a in areas if a["name"] == selected_name)

# Sample ATES
classes = sample_ates(area["tif"], points)

# Metrics
valid_cls = [c for c in classes if c > 0]
col1, col2, col3, col4, col5 = st.columns(5)
total_pts = len(valid_cls) or 1
col1.metric("Points on route", len(points))
col2.metric("Max ATES", max(valid_cls) if valid_cls else "—")

total_dist_km = _cumulative_distance(points)[-1] / 1000
col3.metric("Total distance", f"{total_dist_km:.1f} km")

n_considerable_high = sum(1 for c in valid_cls if c >= 3)
col4.metric("Considerable/High pts", n_considerable_high)

ele_values = [p["ele"] for p in points if p.get("ele") is not None]
if ele_values:
    col5.metric("Elevation range", f"{min(ele_values):.0f}–{max(ele_values):.0f} m")

st.markdown(f"**Area:** {area['name']}")

# Class breakdown
if valid_cls:
    st.markdown("#### ATES class breakdown")
    breakdown_cols = st.columns(4)
    for i, cls in enumerate(range(1, 5)):
        pct = 100 * sum(1 for c in valid_cls if c == cls) / total_pts
        breakdown_cols[i].metric(ATES_LABELS[cls], f"{pct:.0f}%")

# Map
m = _build_route_map(area, points, classes)
st_folium(m, use_container_width=True, height=550)

# Elevation profile
chart = _elevation_profile(points, classes)
if chart:
    st.altair_chart(chart, use_container_width=True)
