from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from ates.validate import Validator

RES = 30.0
TRANSFORM = Affine(RES, 0, 0, 0, -RES, 0)
SHAPE = (20, 20)


def _flat_dem():
    return np.full(SHAPE, 500.0, dtype=np.float32)


def _mock_zones():
    """Return a minimal GeoDataFrame with the interface Validator uses."""
    import geopandas as gpd
    from shapely.geometry import box

    zones = gpd.GeoDataFrame(
        {
            "Name": ["Simple", "Challenging", "Complex"],
            "ates_class": [1, 2, 3],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        crs="EPSG:4326",
    )
    return zones


def _make_validator(tmp_path):
    """Build a Validator with all external I/O mocked out."""
    kmz = tmp_path / "layers.kmz"
    kmz.touch()

    model_fn = MagicMock(return_value=np.ones(SHAPE, dtype=np.int16))

    v = Validator(kmz, model_fn)
    return v, model_fn


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_kmz_path_stored_as_path(tmp_path):
    kmz = tmp_path / "layers.kmz"
    kmz.touch()
    v = Validator(str(kmz), lambda d, t: d)
    assert isinstance(v.kmz_path, Path)


def test_model_module_unwrapped(tmp_path):
    kmz = tmp_path / "layers.kmz"
    kmz.touch()
    from ates.models import simple

    v = Validator(kmz, simple)
    assert v.model is simple.run


def test_model_callable_stored_directly(tmp_path):
    kmz = tmp_path / "layers.kmz"
    kmz.touch()
    fn = lambda d, t: d
    v = Validator(kmz, fn)
    assert v.model is fn


# ---------------------------------------------------------------------------
# properties raise before run()
# ---------------------------------------------------------------------------


def test_predicted_raises_before_run(tmp_path):
    v, _ = _make_validator(tmp_path)
    with pytest.raises(RuntimeError):
        _ = v.predicted


def test_truth_raises_before_run(tmp_path):
    v, _ = _make_validator(tmp_path)
    with pytest.raises(RuntimeError):
        _ = v.truth


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_returns_self(tmp_path):
    v, _ = _make_validator(tmp_path)
    zones = _mock_zones()

    with (
        patch("ates.validate.Validator._load_zones", return_value=zones),
        patch(
            "ates.validate.fetch_dem_wcs",
            return_value=(_flat_dem(), (49.0, -123.0, 50.0, -122.0)),
        ),
    ):
        result = v.run()

    assert result is v


def test_run_collapses_extreme_to_complex(tmp_path):
    kmz = tmp_path / "layers.kmz"
    kmz.touch()
    # Model returns class 4 everywhere
    model_fn = MagicMock(return_value=np.full(SHAPE, 4, dtype=np.int16))
    v = Validator(kmz, model_fn)
    zones = _mock_zones()

    with (
        patch("ates.validate.Validator._load_zones", return_value=zones),
        patch(
            "ates.validate.fetch_dem_wcs",
            return_value=(_flat_dem(), (49.0, -123.0, 50.0, -122.0)),
        ),
    ):
        v.run()

    assert (v.predicted[v.predicted != -9999] == 3).all()


def test_run_populates_predicted_and_truth(tmp_path):
    v, _ = _make_validator(tmp_path)
    zones = _mock_zones()

    with (
        patch("ates.validate.Validator._load_zones", return_value=zones),
        patch(
            "ates.validate.fetch_dem_wcs",
            return_value=(_flat_dem(), (49.0, -123.0, 50.0, -122.0)),
        ),
    ):
        v.run()

    assert v.predicted is not None
    assert v.truth is not None
    assert v.predicted.ndim == 2
    assert v.truth.ndim == 2
