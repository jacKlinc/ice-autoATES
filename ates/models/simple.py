"""Simple slope-based ATES classifier.

Classifies terrain purely by slope angle — a useful baseline before more
sophisticated terrain analysis (PRA, runout, forest) is applied.

Classification thresholds:
    < 25°   → 1  Simple
    25–35°  → 2  Challenging
    35–45°  → 3  Complex
    ≥ 45°   → 4  Extreme
"""
from __future__ import annotations

import numpy as np


def slope_deg(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """Compute slope angle (degrees) from a projected DEM.

    Nodata cells (NaN) in the input are masked in the output.
    Resolution values must be in the same units as the DEM (metres for UTM).
    """
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    dy, dx = np.gradient(filled, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[~np.isfinite(dem)] = np.nan
    return slope


def classify(slope: np.ndarray) -> np.ndarray:
    """Map slope angle array to ATES classes (int16, nodata = -9999)."""
    ates = np.full(slope.shape, -9999, dtype=np.int16)
    valid = np.isfinite(slope)
    ates[valid & (slope < 25)] = 1
    ates[valid & (slope >= 25) & (slope < 35)] = 2
    ates[valid & (slope >= 35) & (slope < 45)] = 3
    ates[valid & (slope >= 45)] = 4
    return ates


def run(dem: np.ndarray, transform, crs=None) -> np.ndarray:
    """Classify a projected DEM into ATES classes.

    Args:
        dem: float32 array in a projected CRS (e.g. UTM), nodata as NaN.
        transform: rasterio Affine transform (provides pixel resolution).
        crs: unused; accepted for interface compatibility with other models.

    Returns:
        int16 array of ATES classes (1–4), nodata = -9999.
    """
    slope = slope_deg(dem, res_x=abs(transform.a), res_y=abs(transform.e))
    return classify(slope)
