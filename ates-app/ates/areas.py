from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "areas"

# Standard Canadian ATES colour scheme
ATES_COLOURS = {
    0: (0, 0, 0, 0),         # No terrain — transparent
    1: (0, 180, 0, 200),     # Low — green
    2: (30, 80, 255, 200),   # Moderate — blue
    3: (40, 40, 40, 200),    # Considerable — black
    4: (220, 30, 30, 200),   # High — red
}

ATES_LABELS = {
    0: "No avalanche terrain",
    1: "Low",
    2: "Moderate",
    3: "Considerable",
    4: "High",
}

ATES_HEX = {
    0: "transparent",
    1: "#00b400",
    2: "#1e50ff",
    3: "#282828",
    4: "#dc1e1e",
}


def load_areas() -> list[dict]:
    """
    Return all areas found under data/areas/.
    Each area dict has keys: name, lat, lon, zoom, description, tif.
    """
    areas = []
    for meta_path in sorted(DATA_DIR.glob("*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["tif"] = str(meta_path.parent / "ates_gen.tif")
        areas.append(meta)
    return areas
