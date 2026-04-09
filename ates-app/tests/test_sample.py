import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ates.sample import sample_ates, find_area_for_points


def _make_ates_tif(path: str, values: np.ndarray, bounds, crs="EPSG:4326"):
    """Write a small GeoTIFF with given ATES values in WGS84."""
    height, width = values.shape
    transform = from_bounds(*bounds, width=width, height=height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int16",
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(values.astype("int16"), 1)


@pytest.fixture()
def simple_tif(tmp_path):
    """
    3x3 raster in WGS84 covering (lon=-117 to -116, lat=51 to 52).
    Values: row 0 = [1, 2, 3], row 1 = [2, 3, 4], row 2 = [4, 4, -9999]
    """
    arr = np.array([[1, 2, 3], [2, 3, 4], [4, 4, -9999]], dtype="int16")
    tif = str(tmp_path / "ates.tif")
    _make_ates_tif(tif, arr, bounds=(-117.0, 51.0, -116.0, 52.0))
    return tif


def test_sample_returns_correct_class(simple_tif):
    # Centre of the top-left cell should be ATES 1
    points = [{"lat": 51.833, "lon": -116.833, "ele": 0}]
    result = sample_ates(simple_tif, points)
    assert result == [1]


def test_sample_multiple_points(simple_tif):
    points = [
        {"lat": 51.833, "lon": -116.833, "ele": 0},  # top-left → 1
        {"lat": 51.833, "lon": -116.5, "ele": 0},    # top-mid → 2
        {"lat": 51.833, "lon": -116.167, "ele": 0},  # top-right → 3
    ]
    result = sample_ates(simple_tif, points)
    assert result == [1, 2, 3]


def test_sample_nodata_returns_minus_one(simple_tif):
    # Bottom-right cell has nodata (-9999)
    points = [{"lat": 51.167, "lon": -116.167, "ele": 0}]
    result = sample_ates(simple_tif, points)
    assert result == [-1]


def test_sample_out_of_bounds_returns_minus_one(simple_tif):
    # Point outside the raster extent
    points = [{"lat": 60.0, "lon": -100.0, "ele": 0}]
    result = sample_ates(simple_tif, points)
    assert result == [-1]


def test_find_area_matches_centroid(simple_tif):
    areas = [{"name": "Test Area", "tif": simple_tif, "lat": 51.5, "lon": -116.5}]
    points = [
        {"lat": 51.5, "lon": -116.5, "ele": 0},
        {"lat": 51.6, "lon": -116.6, "ele": 0},
    ]
    result = find_area_for_points(areas, points)
    assert result is not None
    assert result["name"] == "Test Area"


def test_find_area_no_match(simple_tif):
    areas = [{"name": "Test Area", "tif": simple_tif, "lat": 51.5, "lon": -116.5}]
    # Points far outside the raster
    points = [{"lat": 60.0, "lon": -100.0, "ele": 0}]
    result = find_area_for_points(areas, points)
    assert result is None


def test_find_area_empty_points(simple_tif):
    areas = [{"name": "Test Area", "tif": simple_tif, "lat": 51.5, "lon": -116.5}]
    result = find_area_for_points(areas, [])
    assert result is None
