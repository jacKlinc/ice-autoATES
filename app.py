import base64
import io

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import folium
import streamlit as st
from streamlit_folium import st_folium

ROUTES = {
    "Polar Circus": {
        "tif": "data/polar-circus/output_be.tif",
        "lat": 52.13888,
        "lng": -116.98792,
    },
    # "Weeping Wall": {"tif": None, "lat": 52.1543, "long": -117.00535},
}

SLOPE_BOUNDS = [0, 15, 25, 35, 45, 90]
SLOPE_COLOURS = ["none", "green", "blue", "black", "red"]
SLOPE_LABELS = ["< 15° (safe)", "15–25°", "25–35°", "35–45°", "> 45°"]


@st.cache_data
def load_raster(tif_path):
    with rasterio.open(tif_path) as src:
        dem = src.read(1).astype(float)
        res_x, res_y = src.res
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    dy, dx = np.gradient(dem, res_y, res_x)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Reproject slope to EPSG:4326 so the overlay aligns correctly on the map.
    # The source CRS (EPSG:3979, Canada Atlas Lambert) is a conic projection —
    # converting only corner points produces misaligned rectangular bounds.
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
    slope_wgs84 = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=slope_deg.astype(np.float32),
        destination=slope_wgs84,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # Bounds of the reprojected grid in lat/lng
    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * dst_width
    bottom = top + dst_transform.e * dst_height
    latlon_bounds = (bottom, left, top, right)  # (min_lat, min_lon, max_lat, max_lon)

    return dem, slope_deg, slope_wgs84, latlon_bounds


def slope_to_png(slope_deg, alpha):
    a = int(255 * alpha)
    bands = [
        (0, 15, (0, 0, 0, 0)),  # < 15°: transparent
        (15, 25, (0, 180, 0, a)),  # 15-25°: green
        (25, 35, (30, 80, 255, a)),  # 25-35°: blue
        (35, 45, (40, 40, 40, a)),  # 35-45°: dark
        (45, 90, (220, 30, 30, a)),  # 45°+: red
    ]

    rgba = np.zeros((*slope_deg.shape, 4), dtype=np.uint8)
    for lo, hi, colour in bands:
        mask = (slope_deg >= lo) & (slope_deg < hi)
        rgba[mask] = colour

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def build_map(route, slope_wgs84, latlon_bounds, overlay_alpha):
    min_lat, min_lon, max_lat, max_lon = latlon_bounds

    m = folium.Map(
        location=[route["lat"], route["lng"]],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    png = slope_to_png(slope_wgs84, overlay_alpha)
    folium.raster_layers.ImageOverlay(
        image=png,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1.0,
        name="Slope angle",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Centre marker
    folium.Marker(
        location=[route["lat"], route["lng"]],
        tooltip=list(ROUTES.keys())[
            [r["tif"] for r in ROUTES.values()].index(route["tif"])
        ],
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    return m


# --- Sidebar ---
st.set_page_config(page_title="ice-autoATES", layout="wide")
st.sidebar.title("ice-autoATES")

route_name = st.sidebar.selectbox("Route", list(ROUTES.keys()))
route = ROUTES[route_name]

overlay_alpha = st.sidebar.slider("Slope overlay opacity", 0.0, 1.0, 0.6, 0.05)

st.sidebar.markdown("### Slope legend")
for colour, label in zip(SLOPE_COLOURS[1:], SLOPE_LABELS[1:]):
    st.sidebar.markdown(
        f"<span style='color:{colour}'>■</span> {label}", unsafe_allow_html=True
    )
st.sidebar.markdown(f"□ {SLOPE_LABELS[0]}", unsafe_allow_html=True)

# --- Main ---
st.title(route_name)

dem, slope_deg, slope_wgs84, latlon_bounds = load_raster(route["tif"])

col1, col2, col3 = st.columns(3)
col1.metric("Max slope", f"{slope_deg.max():.1f}°")
col2.metric("Mean slope", f"{slope_deg.mean():.1f}°")
col3.metric("Elevation range", f"{dem.min():.0f}–{dem.max():.0f} m")

m = build_map(route, slope_wgs84, latlon_bounds, overlay_alpha)
st_folium(m, use_container_width=True, height=600)
