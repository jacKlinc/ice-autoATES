from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from ates.validate import Validator

SHAPE = (20, 20)
_DEM_BOUNDS = (49.0, -123.0, 50.0, -122.0)  # (min_lat, min_lon, max_lat, max_lon) — BC


def _flat_dem():
    return np.full(SHAPE, 500.0, dtype=np.float32)


def _mock_zones():
    # BC coordinates so estimate_utm_crs returns UTM Zone 10N, matching _DEM_BOUNDS
    return gpd.GeoDataFrame(
        {
            "Name": ["Simple", "Challenging", "Complex"],
            "ates_class": [1, 2, 3],
            "geometry": [
                box(-123.0, 49.0, -122.7, 49.3),
                box(-122.7, 49.0, -122.4, 49.3),
                box(-122.4, 49.0, -122.1, 49.3),
            ],
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kmz(tmp_path):
    path = tmp_path / "layers.kmz"
    path.touch()
    return path


@pytest.fixture
def validator(kmz):
    model_fn = MagicMock(return_value=np.ones(SHAPE, dtype=np.int16))
    return Validator(kmz, model_fn), model_fn


@pytest.fixture
def mocked_run(validator):
    v, model_fn = validator
    with (
        patch("ates.validate.Validator._load_zones", return_value=_mock_zones()),
        patch("ates.validate.fetch_dem_mrdem", return_value=(_flat_dem(), _DEM_BOUNDS)),
    ):
        v.run()
    return v, model_fn


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestInit:
    """Validator.__init__ stores arguments and validates them."""

    def test_kmz_path_stored_as_path(self, kmz):
        v = Validator(str(kmz), lambda d, t: d)
        assert isinstance(v.kmz_path, Path)

    def test_model_module_unwrapped(self, kmz):
        from ates.models import simple

        v = Validator(kmz, simple)
        assert v.model is simple.run

    def test_model_callable_stored_directly(self, kmz):
        fn = lambda d, t: d
        v = Validator(kmz, fn)
        assert v.model is fn

    def test_invalid_dem_source_raises(self, kmz):
        with pytest.raises(ValueError):
            Validator(kmz, lambda d, t: d, dem_source="invalid")


class TestProperties:
    """Lazy properties guard against access before run() is called."""

    @pytest.mark.parametrize("prop", ["predicted", "truth"])
    def test_raises_before_run(self, validator, prop):
        v, _ = validator
        with pytest.raises(RuntimeError):
            getattr(v, prop)


class TestRun:
    """Validator.run() executes the full pipeline with external I/O mocked."""

    def test_returns_self(self, validator):
        v, _ = validator
        with (
            patch("ates.validate.Validator._load_zones", return_value=_mock_zones()),
            patch(
                "ates.validate.fetch_dem_mrdem", return_value=(_flat_dem(), _DEM_BOUNDS)
            ),
        ):
            result = v.run()
        assert result is v

    def test_populates_predicted_and_truth(self, mocked_run):
        v, _ = mocked_run
        assert v.predicted.ndim == 2
        assert v.truth.ndim == 2

    def test_collapses_extreme_to_complex(self, kmz):
        model_fn = MagicMock(return_value=np.full(SHAPE, 4, dtype=np.int16))
        v = Validator(kmz, model_fn)
        with (
            patch("ates.validate.Validator._load_zones", return_value=_mock_zones()),
            patch(
                "ates.validate.fetch_dem_mrdem", return_value=(_flat_dem(), _DEM_BOUNDS)
            ),
        ):
            v.run()
        assert (v.predicted[v.predicted != -9999] == 4).sum() == 0
