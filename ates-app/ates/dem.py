from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from skimage.measure import find_contours

# CDEM Cloud-Optimised GeoTIFF — covers all of Canada, ~30 m resolution.
# Windowed reads via HTTP range requests mean only the requested bbox is downloaded.
_CDEM_COG_URL = (
    "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com"
    "/store/elevation/cdem-cdsm/cdem/cdem-canada-dem.tif"
)


def fetch_dem_wcs(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fetch CDEM elevation for the given WGS84 bounding box via COG windowed read.

    Returns (dem, latlon_bounds) where latlon_bounds = (min_lat, min_lon, max_lat, max_lon)
    in EPSG:4326, and dem is a float32 array with nodata as NaN.
    """
    wgs84 = CRS.from_epsg(4326)

    with rasterio.open(_CDEM_COG_URL) as src:
        native_bounds = transform_bounds(
            wgs84, src.crs, min_lon, min_lat, max_lon, max_lat
        )
        window = src.window(*native_bounds)
        dem = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata
        native_transform = src.window_transform(window)
        crs = src.crs

    if nodata is not None:
        dem[dem == nodata] = np.nan

    if crs.to_epsg() != 4326:
        height, width = dem.shape
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs,
            wgs84,
            width,
            height,
            left=native_transform.c,
            bottom=native_transform.f + native_transform.e * height,
            right=native_transform.c + native_transform.a * width,
            top=native_transform.f,
        )
        dem_wgs84 = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=dem,
            destination=dem_wgs84,
            src_transform=native_transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=wgs84,
            resampling=Resampling.bilinear,
        )
        native_transform = dst_transform
        dem = dem_wgs84

    left = native_transform.c
    top = native_transform.f
    right = left + native_transform.a * dem.shape[1]
    bottom = top + native_transform.e * dem.shape[0]

    return dem, (bottom, left, top, right)


def dem_to_contour_geojson(
    dem: np.ndarray,
    latlon_bounds: tuple[float, float, float, float],
    interval: int = 50,
) -> dict:
    """Extract contour lines from a WGS84 DEM array.

    Returns a GeoJSON FeatureCollection of LineStrings, each with an
    ``elevation`` property (metres).
    """
    min_lat, min_lon, max_lat, max_lon = latlon_bounds
    height, width = dem.shape

    valid = dem[np.isfinite(dem)]
    if valid.size == 0:
        return {"type": "FeatureCollection", "features": []}

    elev_min = int(np.floor(valid.min() / interval) * interval)
    elev_max = int(np.ceil(valid.max() / interval) * interval)
    levels = range(elev_min, elev_max + interval, interval)

    # Replace NaN with a sentinel below every level so find_contours doesn't
    # produce artefacts at nodata boundaries.
    sentinel = float(elev_min - interval)
    dem_filled = np.where(np.isfinite(dem), dem, sentinel)

    features = []
    for level in levels:
        for contour in find_contours(dem_filled, level):
            # contour rows/cols → lon/lat
            coords = [
                [
                    min_lon + (col / width) * (max_lon - min_lon),
                    max_lat - (row / height) * (max_lat - min_lat),
                ]
                for row, col in contour
            ]
            if len(coords) < 2:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"elevation": level},
                }
            )

    return {"type": "FeatureCollection", "features": features}
