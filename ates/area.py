from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from pydantic import BaseModel, model_validator
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from skimage.measure import find_contours

# pysheds 0.5 calls np.in1d which was removed in NumPy 2.0
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, **kw: np.isin(ar1, ar2, **kw).ravel()

from pysheds.grid import Grid

_MRDEM_DTM_VRT = (
    "https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.vrt"
)


class BoundaryBox(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    # Runs "after" Pydantic's validation. "plain" stops running after object returns
    @model_validator(mode="after")
    def check_ordering(self):
        # pylint: disable=E1101
        if self.min_lat > self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon > self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return self

    def buffer(self, deg: float = 0.05):
        # pylint: disable=E1101
        self.min_lat -= deg
        self.max_lat += deg
        self.min_lon -= deg
        self.max_lon += deg

        return self

    def to_tuple(self) -> Tuple[float, float, float, float]:
        # pylint: disable=E1101
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)  # type: ignore


class AvalancheArea(BaseModel):
    name: str
    bbox: BoundaryBox

    def download_box(self, wgs84):
        min_lat, max_lat, min_lon, max_lon = self.bbox.to_tuple()

        with rasterio.open(_MRDEM_DTM_VRT) as src:
            native_bounds = transform_bounds(
                wgs84, src.crs, min_lon, min_lat, max_lon, max_lat
            )
            window = src.window(*native_bounds)
            dem = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
            native_transform = src.window_transform(window)
            crs = src.crs

        return nodata, dem, native_transform, crs

    def download_mrdem(self) -> Tuple[np.ndarray, np.ndarray]:
        wgs84 = CRS.from_epsg(4326)

        # nodata, dem, native_transform, crs = self.download_box(wgs84)
        min_lat, max_lat, min_lon, max_lon = self.bbox.to_tuple()

        with rasterio.open(_MRDEM_DTM_VRT) as src:
            native_bounds = transform_bounds(
                wgs84, src.crs, min_lon, min_lat, max_lon, max_lat
            )
            window = src.window(*native_bounds)
            dem = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
            native_transform = src.window_transform(window)
            crs = src.crs

        # TODO: make transform function
        if nodata is not None:
            dem[dem == nodata] = np.nan

        # MRDEM native CRS is EPSG:3979 (Canada Lambert) — reproject to WGS84
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

        # TODO: explain
        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_transform.a * dst_width
        bottom = top + dst_transform.e * dst_height

        return dem_wgs84, (bottom, left, top, right)

    def generate_d8_pra(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grid = Grid.from_raster(dem)
        dem_conditioned = grid.fill_pits(dem)
        return grid.flowdir(dem_conditioned)


class AvalancheDataset(BaseModel):
    areas: List[AvalancheArea]
    out_dir: Path

    def build(self) -> pd.DataFrame:
        pixels = []
        for area in self.areas:
            dem = area.download_mrdem()
            pra = area.generate_d8_pra(dem)
            # extract features
            pixels.append([dem, pra])
        return pd.DataFrame(pixels)
