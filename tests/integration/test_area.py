from pathlib import Path
import json

import pytest
import numpy as np

from ates.area import BoundaryBox, AvalancheArea, AvalancheDataset

_SEYMOUR_BBOX = BoundaryBox(
    min_lat=49.351913, max_lat=49.440163, min_lon=-122.975281, max_lon=-122.906245
)

_CACHE_DIR = Path("tests/data/seymour")


def test_avalanche_dataset():
    bb = BoundaryBox(min_lat=2, max_lat=2, min_lon=3, max_lon=4)
    aa = AvalancheArea(name="Test", bbox=bb)
    AvalancheDataset(areas=[aa], out_dir=Path(__file__))


@pytest.fixture(scope="session")
def seymour_dem():
    dem_path = _CACHE_DIR / "dem.npy"
    bounds_path = _CACHE_DIR / "bounds.json"

    if dem_path.exists() and dem_path.stat().st_size > 0:
        return np.load(dem_path), tuple(json.loads(bounds_path.read_text()))

    area = AvalancheArea(name="mount-seymour", bbox=_SEYMOUR_BBOX)
    dem, bounds = area.download_mrdem()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(dem_path, dem)
    bounds_path.write_text(json.dumps(list(bounds)))
    return dem, bounds


def test_download_mrdem(seymour_dem):
    dem_wgs84, (bottom, left, top, right) = seymour_dem

    # Simple matrix size check
    assert dem_wgs84 is not None
    assert dem_wgs84.ndim == 2
    assert dem_wgs84.shape[0] > 0 and dem_wgs84.shape[1] > 0
    assert np.isfinite(dem_wgs84).any()
    # Check bounds are valid
    assert bottom < top
    assert left < right
