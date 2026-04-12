import numpy as np
import pytest
from affine import Affine

from ates.models.simple import classify, run, slope_deg

RES = 30.0  # metres — typical UTM pixel size
TRANSFORM = Affine(RES, 0, 0, 0, -RES, 0)


# ---------------------------------------------------------------------------
# slope_deg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elevation", [0.0, 100.0, 2000.0])
def test_slope_flat_dem_is_zero(elevation):
    dem = np.full((5, 5), elevation, dtype=np.float32)
    slope = slope_deg(dem, res_x=RES, res_y=RES)
    np.testing.assert_allclose(slope, 0.0, atol=1e-5)


@pytest.mark.parametrize("nodata_pos", [(0, 0), (2, 2), (4, 4)])
def test_slope_nodata_propagates(nodata_pos):
    dem = np.full((5, 5), 100.0, dtype=np.float32)
    dem[nodata_pos] = np.nan
    slope = slope_deg(dem, res_x=RES, res_y=RES)
    assert np.isnan(slope[nodata_pos])


@pytest.mark.parametrize("rise_per_pixel, res", [
    (1.0, 30.0),
    (2.0, 10.0),
    (0.5, 50.0),
])
def test_slope_inclined_plane(rise_per_pixel, res):
    rows = np.mgrid[0:10, 0:10][0].astype(np.float32) * rise_per_pixel
    slope = slope_deg(rows, res_x=res, res_y=res)
    expected = np.degrees(np.arctan(rise_per_pixel / res))
    np.testing.assert_allclose(slope[1:-1, 1:-1], expected, atol=0.1)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle, expected_class", [
    (0.0,  1),   # flat → Simple
    (24.9, 1),   # just below threshold → Simple
    (25.0, 2),   # threshold → Challenging
    (30.0, 2),   # mid Challenging
    (35.0, 3),   # threshold → Complex
    (44.9, 3),   # just below threshold → Complex
    (45.0, 4),   # threshold → Extreme
    (60.0, 4),   # steep → Extreme
])
def test_classify_thresholds(angle, expected_class):
    result = classify(np.array([[angle]], dtype=np.float32))
    assert result[0, 0] == expected_class


@pytest.mark.parametrize("angle", [np.nan, float("nan")])
def test_classify_nan_is_nodata(angle):
    result = classify(np.array([[angle]], dtype=np.float32))
    assert result[0, 0] == -9999


def test_classify_output_dtype():
    assert classify(np.zeros((3, 3), dtype=np.float32)).dtype == np.int16


def test_classify_all_classes_present():
    slope = np.array([[10.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    assert set(classify(slope)[0]) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(5, 5), (10, 20), (3, 100)])
def test_run_preserves_shape(shape):
    dem = np.full(shape, 500.0, dtype=np.float32)
    assert run(dem, TRANSFORM).shape == shape


@pytest.mark.parametrize("elevation", [0.0, 500.0, 1500.0])
def test_run_flat_dem_all_simple(elevation):
    dem = np.full((10, 10), elevation, dtype=np.float32)
    result = run(dem, TRANSFORM)
    assert (result[result != -9999] == 1).all()


@pytest.mark.parametrize("nodata_pos", [(0, 0), (2, 2)])
def test_run_nodata_passthrough(nodata_pos):
    dem = np.full((5, 5), 500.0, dtype=np.float32)
    dem[nodata_pos] = np.nan
    assert run(dem, TRANSFORM)[nodata_pos] == -9999
