from pathlib import Path
from ates.area import BoundaryBox, AvalancheArea, AvalancheDataset

def test_avalanche_dataset():
    bb = BoundaryBox(min_lat=2, max_lat=2, min_lon=3, max_lon=4)
    aa = AvalancheArea(name="Test", bbox=bb)
    AvalancheDataset(areas=[aa], out_dir=Path(__file__))
