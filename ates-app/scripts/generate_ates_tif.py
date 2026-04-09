"""Generate ates_gen.tif for an area from the CDEM.

Usage (from ates-app/):
    uv run --group dev python scripts/generate_ates_tif.py <area-dir-name>

Example:
    uv run python scripts/generate_ates_tif.py internation-mountain

The script reads data/areas/<name>/metadata.json, fetches the CDEM for a
buffered bounding box around the lat/lon centre, classifies terrain into
ATES classes by slope angle, and writes data/areas/<name>/ates_gen.tif.

Slope-based ATES classification (simplified AutoATES v2 rules):
    < 25°   → 1  Simple
    25–35°  → 2  Challenging
    35–45°  → 3  Complex
    ≥ 45°   → 4  Extreme
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

# Buffer around the metadata centre point (degrees).
# ~0.05° lat ≈ 5.5 km, ~0.08° lon ≈ 5.5 km at 51°N.
_LAT_BUFFER = 0.05
_LON_BUFFER = 0.08

# Target CRS for slope calculation — UTM Zone 11N covers BC/Alberta.
# Adjust to Zone 10N (32610) for areas west of 120°W.
_UTM_CRS = CRS.from_epsg(32611)

_CDEM_COG_URL = (
    "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com"
    "/store/elevation/cdem-cdsm/cdem/cdem-canada-dem.tif"
)

_DATA_DIR = Path(__file__).parent.parent / "data" / "areas"


def _utm_zone_crs(lon: float) -> CRS:
    """Return the UTM CRS for a given longitude."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone  # northern hemisphere
    return CRS.from_epsg(epsg)


def _fetch_cdem_utm(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, utm_crs: CRS
) -> tuple[np.ndarray, rasterio.Affine]:
    """Download the CDEM window covering the bbox and reproject to UTM."""
    wgs84 = CRS.from_epsg(4326)

    with rasterio.open(_CDEM_COG_URL) as src:
        native_bounds = transform_bounds(wgs84, src.crs, min_lon, min_lat, max_lon, max_lat)
        window = src.window(*native_bounds)
        dem_native = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata
        native_transform = src.window_transform(window)
        native_crs = src.crs

    if nodata is not None:
        dem_native[dem_native == nodata] = np.nan

    h, w = dem_native.shape
    dst_transform, dst_w, dst_h = calculate_default_transform(
        native_crs, utm_crs, w, h,
        left=native_transform.c,
        bottom=native_transform.f + native_transform.e * h,
        right=native_transform.c + native_transform.a * w,
        top=native_transform.f,
    )
    dem_utm = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=dem_native,
        destination=dem_utm,
        src_transform=native_transform,
        src_crs=native_crs,
        dst_transform=dst_transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear,
    )
    return dem_utm, dst_transform


def _slope_deg(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    dy, dx = np.gradient(filled, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[~np.isfinite(dem)] = np.nan
    return slope


def _classify(slope: np.ndarray) -> np.ndarray:
    """Map slope angle to ATES class (1–4); nodata → -9999."""
    ates = np.full(slope.shape, -9999, dtype=np.int16)
    valid = np.isfinite(slope)
    ates[valid & (slope < 25)] = 1
    ates[valid & (slope >= 25) & (slope < 35)] = 2
    ates[valid & (slope >= 35) & (slope < 45)] = 3
    ates[valid & (slope >= 45)] = 4
    return ates


def generate(area_name: str) -> Path:
    area_dir = _DATA_DIR / area_name
    meta_path = area_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    lat, lon = meta["lat"], meta["lon"]
    utm_crs = _utm_zone_crs(lon)

    min_lat = lat - _LAT_BUFFER
    max_lat = lat + _LAT_BUFFER
    min_lon = lon - _LON_BUFFER
    max_lon = lon + _LON_BUFFER

    print(f"Fetching CDEM for {meta['name']} …")
    dem, transform = _fetch_cdem_utm(min_lat, min_lon, max_lat, max_lon, utm_crs)
    print(f"  DEM shape: {dem.shape}, resolution: {transform.a:.0f} m")

    slope = _slope_deg(dem, res_x=transform.a, res_y=abs(transform.e))
    ates = _classify(slope)

    out_path = area_dir / "ates_gen.tif"
    h, w = ates.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=np.int16,
        crs=utm_crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(ates, 1)

    valid = ates[ates != -9999]
    labels = {1: "Simple", 2: "Challenging", 3: "Complex", 4: "Extreme"}
    print(f"Saved {out_path}")
    for cls in range(1, 5):
        pct = 100 * (valid == cls).sum() / len(valid) if len(valid) else 0
        print(f"  Class {cls} {labels[cls]:12s}: {pct:.1f}%")

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    generate(sys.argv[1])
