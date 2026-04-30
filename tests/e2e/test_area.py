from pathlib import Path
import json

from rasterio.crs import CRS
import pytest
import numpy as np
from affine import Affine

from ates.area import BoundaryBox, AvalancheArea, AvalancheDataset

AREA_MAP = [
    {
        "name": "mount-seymour",
        "bbox": BoundaryBox(
            min_lat=49.334422724929645,
            max_lat=49.4575084323684,
            min_lon=-123.02740693100154,
            max_lon=-122.85407421008304,
        ),
    }
] * 10


def test_avalanche_dataset():
    # 9.05s for 10 examples

    areas = [AvalancheArea(name=a["name"], bbox=a["bbox"]) for a in AREA_MAP]
    d = AvalancheDataset(
        areas=areas, out_dir=Path(__file__), dst_crs=CRS.from_epsg(4326)
    )
    dataset = d.build()
    assert True


@pytest.skip()
def test_avalanche_dataset_sync():
    # 6.89s for 10 examples
    # 
    areas = [AvalancheArea(name=a["name"], bbox=a["bbox"]) for a in AREA_MAP]
    d = AvalancheDataset(
        areas=areas, out_dir=Path(__file__), dst_crs=CRS.from_epsg(4326)
    )
    dst_crs = CRS.from_epsg(4326)
    for area in d.areas:
        area._download_box(dst_crs)
    assert True
