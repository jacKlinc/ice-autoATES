# ATES Model Benchmark Suite

## Context
Build a benchmark comparison suite for three ATES terrain classification models:
- **Simple** — slope-only (already exists in `generate_ates_tif.py`)
- **AutoATES v1** — PRA + alpha-angle runout + binary forest (no overhead exposure)
- **AutoATES v2** — extends v1 with Cauchy fuzzy PRA, wind shelter, forest density, Flow-Py runout, overhead exposure, island filter

Test areas: **Bow Summit** and **Connaught Creek**.
Output: per-class and macro F1 scores formatted like the paper (Vors et al. 2024).

---

## Ground Truth Problem
The paper uses expert consensus maps — not publicly available. User only has the paper image.

**Strategy:**
1. Design framework to accept any `ates_benchmark.tif` dropped into `benchmark/ground_truth/<area>/`
2. The official AutoATES v2 repo (`AutoATES/AutoATES-v2.0`) includes Bow Summit **input** data (`dem.tif`, `forest.tif`, `pra.tif`, `z_delta.tif`) + v2 model output `outputs/ates_gen.tif` — download these as interim inputs for Bow Summit
3. For true ground truth: contact Vors et al. authors, or use Avalanche Canada published ATES polygons as a proxy

---

## New Files

```
ates-app/
├── ates/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── simple.py        # thin wrapper — reuses _classify/_slope_deg from generate_ates_tif.py
│   │   ├── v1.py            # PRA (slope Cauchy) + D8 runout + binary forest
│   │   └── v2.py            # full AutoATES v2 (Cauchy PRA + wind shelter + Flow-Py + forest density + overhead)
│   └── evaluate.py          # confusion matrix, per-class F1, macro F1 (no sklearn needed)
├── benchmark/
│   ├── areas.py             # Bow Summit + Connaught Creek bbox + metadata
│   ├── ground_truth/        # Drop ates_benchmark.tif here per area
│   │   ├── bow-summit/
│   │   └── connaught-creek/
│   └── run.py               # CLI: fetch DEM+forest → run all models → score → print table
└── tests/
    └── test_evaluate.py     # unit tests: F1 edge cases (empty class, perfect score, etc.)
```

**Critical files to modify:**
- `ates-app/pyproject.toml` — add `avaframe` dependency (provides Flow-Py for v2 runout)

---

## Common Model Interface
All three models in `ates/models/` expose the same function:
```python
def run(dem: np.ndarray, transform: Affine, crs: CRS, **kwargs) -> np.ndarray:
    """Returns int16 ATES raster (classes 1-4, nodata=-9999) in same grid/CRS as input."""
```

---

## Algorithm Details

### Simple (`models/simple.py`)
Refactor `_classify` and `_slope_deg` from `ates-app/scripts/generate_ates_tif.py` into a shared location.
- `< 25°` → 1, `25–35°` → 2, `35–45°` → 3, `≥ 45°` → 4

### AutoATES v1 (`models/v1.py`)
Port of the ArcGIS/TauDEM approach to pure Python (TauDEM replaced with numpy D8).
1. **Slope classification**: SAT01=15°, SAT12=18°, SAT23=28°, SAT34=39° (smoothed for class 4)
2. **PRA**: Veitinger Cauchy on slope only → binary mask (threshold 0.15)
3. **Binary forest**: NALCMS cells classified as forest zero out PRA
4. **D8 runout**: steepest descent from each PRA cell, stop when alpha angle > AAT3=33°
5. Classify runout cells as class 2 (challenging); no overhead exposure in v1

### AutoATES v2 (`models/v2.py`)
From official source at `AutoATES/AutoATES-v2.0`.
1. **PRA** — three-factor Cauchy fuzzy (threshold 0.15):
   - Slope: `a=11, b=4, c=43`
   - Wind shelter: `a=3, b=10, c=3` (circular sector, 250m radius, 0.5 quantile)
   - Forest: `a=40, b=3.5, c=-15` (PCC type; forest reduces PRA membership)
   - Combined: `minvar*(1−minvar) + minvar*(slope+wind+forest)/3`
2. **Slope classification**: same thresholds as v1 (SAT01=15, SAT12=18, SAT23=28, SAT34=39)
3. **Flow-Py runout** (via `avaframe`): alpha angles AAT1=18°, AAT2=24°, AAT3=33°
4. **Overhead exposure**: upstream cell count reclassified with CC1=5, CC2=40
5. **Layer combine**: `max(slope_class, runout_class, overhead_class)`
6. **Forest density adjustment** (4 categories, values 10/20/30/40) via lookup table
7. **Island filter**: remove patches < ISL_SIZE=30,000 m², fill gaps by interpolation

### Forest Data Source
NALCMS 2020 (North American Land Change Monitoring System) at 30m — publicly available COG from CEC. Similar windowed-read approach to `dem.py`. Add `fetch_forest_nalcms(min_lat, min_lon, max_lat, max_lon)` to `ates/dem.py`.

---

## Evaluation (`ates/evaluate.py`)
```python
def confusion_matrix(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """4x4 matrix for classes 1-4. Ignores nodata (-9999)."""

def compute_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Returns {'class_f1': {1: float, 2: float, 3: float, 4: float}, 'macro_f1': float,
                'precision': {...}, 'recall': {...}}"""
```
Implement F1 from scratch (no scikit-learn): `f1 = 2*tp / (2*tp + fp + fn)` per class.

---

## Benchmark Areas (`benchmark/areas.py`)
```python
AREAS = {
    "bow-summit": {
        "lat": 51.6976, "lon": -116.4934,
        "min_lat": 51.65, "min_lon": -116.55, "max_lat": 51.75, "max_lon": -116.43,
    },
    "connaught-creek": {
        "lat": 51.259, "lon": -117.631,  # Rogers Pass, BC
        "min_lat": 51.21, "min_lon": -117.70, "max_lat": 51.31, "max_lon": -117.56,
    },
}
```

---

## Benchmark Runner (`benchmark/run.py`)
```
$ uv run python benchmark/run.py

Area: Bow Summit
─────────────────────────────────────────────────────────────────────
Model        | Simple F1 | Chall. F1 | Complex F1 | Extreme F1 | Macro F1
─────────────────────────────────────────────────────────────────────
Simple       |   0.XX    |   0.XX    |   0.XX     |   0.XX     |  0.XX
AutoATES v1  |   0.XX    |   0.XX    |   0.XX     |   0.XX     |  0.XX
AutoATES v2  |   0.XX    |   0.XX    |   0.XX     |   0.XX     |  0.XX

[Ground truth not found] — place ates_benchmark.tif in benchmark/ground_truth/connaught-creek/
```
Saves model outputs to `benchmark/outputs/<area>/<model>.tif` for visual inspection.

---

## Dependencies to Add
- `avaframe` → `ates-app/pyproject.toml` main dependencies (provides `flowpy`)

---

## Implementation Order
1. `ates/evaluate.py` + `tests/test_evaluate.py` — metrics first, test-driven
2. `benchmark/areas.py` — bbox definitions
3. `ates/models/simple.py` — refactor existing code
4. `ates/models/v1.py` — PRA + D8 runout
5. `ates/models/v2.py` — full v2
6. `fetch_forest_nalcms()` in `ates/dem.py`
7. `benchmark/run.py` — wire everything together

---

## Verification
1. `uv run --group dev pytest tests/test_evaluate.py` — metrics unit tests pass
2. `uv run python benchmark/run.py` — runs without error, saves output TIFs to `benchmark/outputs/`
3. Load output TIFs in the Area Map Streamlit page to visually compare model outputs
4. Once ground truth is available: macro F1 should approach paper values (Bow Summit v2 ≈ 0.77, v1 ≈ 0.64)

---

## References
- Vors et al. (2024) — AutoATES v2: https://nhess.copernicus.org/articles/24/1779/2024/nhess-24-1779-2024.html
- AutoATES v2 source code: https://github.com/AutoATES/AutoATES-v2.0
- AutoATES v1 source code: https://github.com/hvtola/AutoATES_v1.0
- Veitinger et al. (2016) — PRA model: https://tc.copernicus.org/articles/10/1699/2016/
- NALCMS 2020 forest data: http://www.cec.org/north-american-land-change-monitoring-system/
