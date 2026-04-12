"""Validation pipeline for ATES classification models against Avalanche Canada KMZ data."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ates.dem import fetch_dem_wcs

_NAME_TO_CLASS = {"Simple": 1, "Challenging": 2, "Complex": 3, "Extreme": 4}

# Model callable type: (dem: np.ndarray, transform: Affine) -> np.ndarray
ModelFn = Callable[[np.ndarray, object], np.ndarray]


class Validator:
    """Validates an ATES model against ground-truth zones from a KMZ file.

    Args:
        kmz: Path to a KMZ file.
        model: Module or callable with signature ``run(dem, transform) -> np.ndarray``.
               The returned array must use int16 ATES classes (1–4) with nodata = -9999.

    Example::

        from ates import evaluate
        from ates.models import simple
        from ates.validate import Validator

        v = Validator("data/areas/mount-seymour/layers.kmz", simple).run()
        evaluate.report(v.predicted, v.truth)
        evaluate.plot_confusion_matrix(v.predicted, v.truth)
        evaluate.plot_side_by_side(v.predicted, v.truth)
    """

    def __init__(self, kmz: Path | str, model: ModuleType | ModelFn) -> None:
        self.kmz_path: Path = Path(kmz)
        self.model: ModelFn = model.run if isinstance(model, ModuleType) else model

        self._predicted: np.ndarray | None = None
        self._truth: np.ndarray | None = None

    @property
    def predicted(self) -> np.ndarray:
        self._require_run()
        return self._predicted

    @property
    def truth(self) -> np.ndarray:
        self._require_run()
        return self._truth

    def run(self) -> "Validator":
        """Execute the full pipeline: KMZ → DEM → model → rasterised truth."""
        zones = self._load_zones()
        utm_crs = zones.estimate_utm_crs()

        dem_utm, dst_transform, dst_width, dst_height = self._fetch_and_reproject_dem(
            zones, utm_crs
        )

        predicted = self.model(dem_utm, dst_transform)
        # Collapse Extreme (4) → Complex (3): Avalanche Canada KMZs use ATES v1
        # which has no Extreme class.
        predicted[predicted == 4] = 3

        self._predicted = predicted
        self._truth = self._rasterize_truth(
            zones, utm_crs, dst_transform, dst_height, dst_width
        )
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_zones(self) -> gpd.GeoDataFrame:
        zones = gpd.read_file(self.kmz_path, driver="libkml", layer="Zones")
        zones["ates_class"] = zones["Name"].map(_NAME_TO_CLASS)
        return zones

    def _fetch_and_reproject_dem(
        self, zones: gpd.GeoDataFrame, utm_crs: CRS
    ) -> tuple[np.ndarray, object, int, int]:
        wgs84 = CRS.from_epsg(4326)
        minx, miny, maxx, maxy = zones.total_bounds

        dem_wgs84, (min_lat, min_lon, max_lat, max_lon) = fetch_dem_wcs(
            min_lat=miny, min_lon=minx, max_lat=maxy, max_lon=maxx
        )
        src_transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat, dem_wgs84.shape[1], dem_wgs84.shape[0]
        )
        dst_transform, dst_width, dst_height = calculate_default_transform(
            wgs84,
            utm_crs,
            dem_wgs84.shape[1],
            dem_wgs84.shape[0],
            left=min_lon,
            bottom=min_lat,
            right=max_lon,
            top=max_lat,
        )
        dem_utm = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=dem_wgs84,
            destination=dem_utm,
            src_transform=src_transform,
            src_crs=wgs84,
            dst_transform=dst_transform,
            dst_crs=utm_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return dem_utm, dst_transform, dst_width, dst_height

    def _rasterize_truth(
        self,
        zones: gpd.GeoDataFrame,
        utm_crs: CRS,
        dst_transform: object,
        dst_height: int,
        dst_width: int,
    ) -> np.ndarray:
        zones_utm = zones.to_crs(utm_crs)
        return rasterize(
            (
                (geom, val)
                for geom, val in zip(zones_utm.geometry, zones_utm["ates_class"])
                if geom is not None
            ),
            out_shape=(dst_height, dst_width),
            transform=dst_transform,
            fill=-9999,
            dtype=np.int16,
            merge_alg=rasterio.enums.MergeAlg.replace,
        )

    def _require_run(self) -> None:
        if self._predicted is None:
            raise RuntimeError("Call .run() before accessing results.")
