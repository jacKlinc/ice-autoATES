import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from ates.area import AvalancheArea

# Small synthetic DEM in EPSG:3979 coords near Mt Seymour
_NATIVE_TRANSFORM = Affine(30.0, 0.0, -1_964_987.0, 0.0, -30.0, 487_201.0)
_SRC_CRS = CRS.from_epsg(3979)
_DST_CRS = CRS.from_epsg(4326)
_NODATA = -32767.0


def _make_dem(rows: int = 10, cols: int = 10, nodata_count: int = 3) -> np.ndarray:
    """Synthetic float32 DEM with optional nodata sentinels in the first column."""
    rng = np.random.default_rng(42)
    dem = rng.uniform(800.0, 1500.0, (rows, cols)).astype(np.float32)
    dem[:nodata_count, 0] = _NODATA
    return dem


class TestTransform:
    """Unit tests for AvalancheArea._transform — no network calls, synthetic DEM only."""

    def test_nodata_replaced_with_nan(self):
        """Nodata sentinel values must not appear in the reprojected output."""
        dem = _make_dem(nodata_count=3)
        _, _, _, _, dst_dem = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert not np.any(
            dst_dem == _NODATA
        ), "nodata sentinel should be replaced by NaN"

    def test_output_is_2d(self):
        """Reprojected DEM must be a 2-D array."""
        dem = _make_dem()
        _, _, _, _, dst_dem = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert dst_dem.ndim == 2

    def test_output_shape_matches_reported_dims(self):
        """dst_dem.shape must equal (dst_height, dst_width) from the same call."""
        dem = _make_dem()
        _, dst_width, dst_height, _, dst_dem = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert dst_dem.shape == (dst_height, dst_width)

    def test_output_has_finite_values(self):
        """At least some cells must survive reprojection as finite values."""
        dem = _make_dem()
        _, _, _, _, dst_dem = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert np.isfinite(dst_dem).any(), "reprojected DEM should contain valid values"

    def test_none_nodata_leaves_values_intact(self):
        """When nodata is None no values should be replaced before reprojection."""
        dem = _make_dem(nodata_count=0)
        _, _, _, _, dst_dem = AvalancheArea._transform(
            None, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert np.isfinite(dst_dem).sum() > 0

    def test_native_transform_returned_unchanged(self):
        """_transform must pass the native transform through unmodified."""
        dem = _make_dem()
        _, _, _, returned_transform, _ = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert returned_transform == _NATIVE_TRANSFORM

    def test_dst_transform_is_affine(self):
        """Returned destination transform must be an Affine instance."""
        dem = _make_dem()
        dst_transform, _, _, _, _ = AvalancheArea._transform(
            _NODATA, dem, _SRC_CRS, _DST_CRS, _NATIVE_TRANSFORM
        )
        assert isinstance(dst_transform, Affine)
