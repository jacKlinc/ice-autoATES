"""AutoATES v2 classifier wrapper.

Delegates to the upstream AutoATES_classifier implementation installed via the
autoates-v2-0 package and returns the result as a NumPy array, matching the
interface expected by the rest of this package.

Unlike simple.run(), this model requires pre-computed raster inputs (canopy,
flow paths, PRA starting zones) in addition to the DEM, so it cannot share the
same two-argument signature. Instead it exposes a run() that accepts file paths
and an optional working directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio


def _import_autoates():
    """Import AutoATES module from the autoates_v2 submodule."""
    import autoates_v2.AutoATES_classifier as _mod
    return _mod


def run(
    dem: np.ndarray | Path | str,
    transform,
    canopy: Path | str,
    cell_count: Path | str,
    fp: Path | str,
    sz: Path | str,
    forest_type: str = "bav",
    wd: Path | str | None = None,
    win_size: int = 3,
    isl_size: int = 30000,
) -> np.ndarray:
    """Run the AutoATES v2 classifier and return the ATES raster as a NumPy array.

    Matches the ``run(dem, transform, **kwargs)`` interface used by Validator,
    accepting the UTM DEM as a NumPy array. The array is written to a temporary
    GeoTIFF because the underlying classifier requires file-based I/O.

    Args:
        dem:         UTM DEM as a float32 NumPy array, or a path to a GeoTIFF.
        transform:   Rasterio Affine transform for the DEM array (ignored when
                     dem is a file path).
        canopy:      Path to canopy/forest raster.
        cell_count:  Path to overhead cell-count raster (z_delta proxy).
        fp:          Path to Flow-Py alpha angle raster.
        sz:          Path to PRA binary starting zone raster.
        forest_type: Canopy metric type — one of 'bav', 'stems', 'pcc', 'sen2cc'.
        wd:          Working directory for intermediate files. Uses a temp
                     directory if not provided.
        win_size:    Neighbourhood window size for Class 4 slope smoothing.
        isl_size:    Minimum cluster size (m²) below which islands are removed.

    Returns:
        int16 array of ATES classes (0–4), nodata = -9999.
    """
    _mod = _import_autoates()
    thresholds = _thresholds_for_forest_type(forest_type)

    wd_path = Path(wd) if wd else Path(tempfile.mkdtemp())
    wd_path.mkdir(parents=True, exist_ok=True)

    # Write numpy DEM to disk if an array was passed
    if isinstance(dem, np.ndarray):
        dem_path = wd_path / "dem.tif"
        with rasterio.open(
            dem_path, "w",
            driver="GTiff", dtype="float32",
            count=1, crs="EPSG:32610",
            width=dem.shape[1], height=dem.shape[0],
            transform=transform,
            nodata=float("nan"),
        ) as dst:
            dst.write(dem, 1)
        dem = dem_path

    # SZ is a module-level global in AutoATES_classifier, not a function parameter
    _mod.SZ = str(Path(sz).resolve())

    _mod.AutoATES(
        str(wd_path),
        str(dem),
        str(canopy),
        str(cell_count),
        str(fp),
        **thresholds,
        ISL_SIZE=isl_size,
        WIN_SIZE=win_size,
    )

    with rasterio.open(wd_path / "ates_gen.tif") as src:
        return src.read(1)


def _thresholds_for_forest_type(forest_type: str) -> dict:
    """Return default slope-angle, alpha-angle, and tree thresholds for a given forest type."""
    tree = {
        "pcc":    {"TREE1": 10, "TREE2": 50, "TREE3": 65},
        "bav":    {"TREE1": 10, "TREE2": 20, "TREE3": 25},
        "stems":  {"TREE1": 100, "TREE2": 250, "TREE3": 500},
        "sen2cc": {"TREE1": 20, "TREE2": 60, "TREE3": 85},
    }
    if forest_type not in tree:
        raise ValueError(f"Unknown forest_type '{forest_type}'. Choose from: {list(tree)}")

    return {
        "SAT01": 15, "SAT12": 18, "SAT23": 28, "SAT34": 39,
        "AAT1": 18, "AAT2": 24, "AAT3": 33,
        "CC1": 5, "CC2": 40,
        **tree[forest_type],
    }
