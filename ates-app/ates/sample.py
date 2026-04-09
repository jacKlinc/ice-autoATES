from __future__ import annotations

import rasterio
from pyproj import Transformer
from rasterio.crs import CRS


def sample_ates(tif_path: str, points: list[dict]) -> list[int]:
    """
    Sample an ates_gen.tif raster at each (lat, lon) point.

    Transforms coordinates from EPSG:4326 to the raster's native CRS, then
    uses rasterio's vectorised sample() for lookup.

    Returns a list of ATES class ints (0–4).  Points outside the raster
    extent are returned as -1.
    """
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), raster_crs, always_xy=True
        )

        coords = [
            transformer.transform(p["lon"], p["lat"]) for p in points
        ]

        # rasterio.sample returns an iterator of 1-element arrays
        results: list[int] = []
        for val_arr in src.sample(coords):
            v = int(val_arr[0])
            # Treat nodata (-9999) and out-of-range as -1
            results.append(v if 0 <= v <= 4 else -1)

    return results


def find_area_for_points(
    areas: list[dict], points: list[dict]
) -> dict | None:
    """
    Return the first area whose ates_gen.tif bounding box (in WGS84)
    contains the centroid of `points`.  Returns None if no match.

    Each area dict must have keys: name, tif, lat, lon (centre), and the
    raster bounds are derived on the fly.
    """
    if not points:
        return None

    centroid_lat = sum(p["lat"] for p in points) / len(points)
    centroid_lon = sum(p["lon"] for p in points) / len(points)

    for area in areas:
        with rasterio.open(area["tif"]) as src:
            raster_crs = src.crs
            bounds = src.bounds

        transformer = Transformer.from_crs(
            raster_crs, CRS.from_epsg(4326), always_xy=True
        )
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

        if min_lat <= centroid_lat <= max_lat and min_lon <= centroid_lon <= max_lon:
            return area

    return None
