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


def build_map(area: dict, ates_wgs84, latlon_bounds, alpha: float):
    min_lat, min_lon, max_lat, max_lon = latlon_bounds
    m = folium.Map(
        location=[area["lat"], area["lon"]],
        zoom_start=area.get("zoom", 13),
        tiles="CartoDB positron",
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
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

m = build_map(area, ates_wgs84, latlon_bounds, alpha)
st_folium(m, use_container_width=True, height=620)
