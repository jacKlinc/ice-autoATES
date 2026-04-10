from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "areas"

# Standard Canadian ATES colour scheme
ATES_COLOURS = {
    0: (0, 0, 0, 0),         # No terrain — transparent
    1: (0, 180, 0, 200),     # Simple — green
    2: (30, 80, 255, 200),   # Challenging — blue
    3: (40, 40, 40, 200),    # Complex — black
    4: (220, 30, 30, 200),   # Extreme — red
}

ATES_LABELS = {
    0: "No avalanche terrain",
    1: "Simple",
    2: "Challenging",
    3: "Complex",
    4: "Extreme",
}

ATES_HEX = {
    0: "transparent",
    1: "#00b400",
    2: "#1e50ff",
    3: "#000000",
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
