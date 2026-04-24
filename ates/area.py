from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from pydantic import BaseModel, model_validator
import pandas as pd

# pysheds 0.5 calls np.in1d which was removed in NumPy 2.0
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, **kw: np.isin(ar1, ar2, **kw).ravel()

from pysheds.grid import Grid
from ates.dem import fetch_dem_mrdem  # TODO: move here to fix imports


class BoundaryBox(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    # Runs "after" Pydantic's validation. "plain" stops running after object returns
    @model_validator(mode="after")
    def check_ordering(self):
        if self.min_lat > self.max_lat:  # type: ignore
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon > self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return self

    def buffer(self, deg: float = 0.05):
        self.min_lat -= deg
        self.max_lat += deg
        self.min_lon -= deg
        self.max_lon += deg  # type: ignore

        return self

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)  # type: ignore


class AvalancheArea(BaseModel):
    name: str
    bbox: BoundaryBox

    def download_mrdem(self) -> Tuple[np.ndarray, np.ndarray]:
        return fetch_dem_mrdem(
            self.bbox.min_lat, self.bbox.min_lon, self.bbox.max_lat, self.bbox.max_lon
        )

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
