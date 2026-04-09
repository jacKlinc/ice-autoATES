from __future__ import annotations

import base64
import io

import folium
import numpy as np
import rasterio
import streamlit as st
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium

from ates.areas import ATES_COLOURS, ATES_HEX, ATES_LABELS, load_areas
from ates.dem import dem_to_contour_geojson, fetch_dem_wcs

st.set_page_config(page_title="Area Map — AutoATES", layout="wide")


@st.cache_data
def load_ates_raster(tif_path: str):
    """Load and reproject an ates_gen.tif to EPSG:4326 for map overlay."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    if nodata is not None:
        data[data == nodata] = np.nan

    dst_crs = "EPSG:4326"
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs,
        dst_crs,
        width,
        height,
        left=transform.c,
        bottom=transform.f + transform.e * height,
        right=transform.c + transform.a * width,
        top=transform.f,
    )
    ates_wgs84 = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=data.astype(np.float32),
        destination=ates_wgs84,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * dst_width
    bottom = top + dst_transform.e * dst_height
    latlon_bounds = (bottom, left, top, right)

    return ates_wgs84, latlon_bounds


@st.cache_data
def load_contours(min_lat: float, min_lon: float, max_lat: float, max_lon: float, interval: int):
    """Fetch DEM and extract contours for the given bounding box."""
    dem, bounds = fetch_dem_wcs(min_lat, min_lon, max_lat, max_lon)
    return dem_to_contour_geojson(dem, bounds, interval=interval)


def ates_to_png(ates: np.ndarray, alpha: float) -> str:
    a = int(255 * alpha)
    rgba = np.zeros((*ates.shape, 4), dtype=np.uint8)
    for cls, (r, g, b, _) in ATES_COLOURS.items():
        if cls == 0:
            continue
        mask = np.isfinite(ates) & (np.round(ates).astype(int) == cls)
        rgba[mask] = (r, g, b, a)

    buf = io.BytesIO()
    import matplotlib.pyplot as plt
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def build_map(area: dict, ates_wgs84, latlon_bounds, alpha: float, contour_geojson: dict | None):
    min_lat, min_lon, max_lat, max_lon = latlon_bounds
    m = folium.Map(
        location=[area["lat"], area["lon"]],
        zoom_start=area.get("zoom", 13),
        tiles="CartoDB positron",
    )

    if contour_geojson is not None:
        folium.GeoJson(
            contour_geojson,
            name="Contours",
            style_function=lambda _: {
                "color": "#555555",
                "weight": 0.8,
                "opacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(fields=["elevation"], aliases=["Elevation (m)"]),
        ).add_to(m)

    png = ates_to_png(ates_wgs84, alpha)
    folium.raster_layers.ImageOverlay(
        image=png,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1.0,
        name="ATES classification",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# --- Sidebar ---
st.sidebar.title("Area Map")
areas = load_areas()
area_names = [a["name"] for a in areas]
selected_name = st.sidebar.selectbox("Area", area_names)
area = next(a for a in areas if a["name"] == selected_name)

alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.7, 0.05)
# CDEM native resolution is ~30 m, so 30 m contours match the data density.
contour_interval = 30

st.sidebar.markdown("### ATES legend")
for cls in range(1, 5):
    hex_col = ATES_HEX[cls]
    label = ATES_LABELS[cls]
    st.sidebar.markdown(
        f"<span style='color:{hex_col}; font-size:1.2em'>■</span> **{cls}** — {label}",
        unsafe_allow_html=True,
    )

if area.get("description"):
    st.sidebar.markdown("---")
    st.sidebar.caption(area["description"])

# --- Main ---
st.title(f"ATES Map — {selected_name}")

ates_wgs84, latlon_bounds = load_ates_raster(area["tif"])

valid = ates_wgs84[np.isfinite(ates_wgs84) & (ates_wgs84 > 0)]
if valid.size > 0:
    counts = {cls: int(np.sum(np.round(valid).astype(int) == cls)) for cls in range(1, 5)}
    total = sum(counts.values())
    cols = st.columns(4)
    for i, cls in enumerate(range(1, 5)):
        pct = 100 * counts[cls] / total if total > 0 else 0
        cols[i].metric(ATES_LABELS[cls], f"{pct:.0f}%")

@st.fragment
def render_map(area, ates_wgs84, latlon_bounds, alpha, contour_interval):
    min_lat, min_lon, max_lat, max_lon = latlon_bounds
    with st.spinner("Fetching elevation data…"):
        try:
            contour_geojson = load_contours(min_lat, min_lon, max_lat, max_lon, contour_interval)
        except Exception as e:
            st.warning(f"Could not load contours: {e}")
            contour_geojson = None

    m = build_map(area, ates_wgs84, latlon_bounds, alpha, contour_geojson)
    st_folium(m, use_container_width=True, height=620, returned_objects=[])


render_map(area, ates_wgs84, latlon_bounds, alpha, contour_interval)
