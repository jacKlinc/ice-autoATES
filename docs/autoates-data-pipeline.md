# AvalancheArea Class + Data Pipeline Plan

## Context

Replace AutoATES v2's slow Flow-Py pipeline with a faster D8-based feature extractor that can run on Lambda.

`autoates_v2/` is **legacy** — all new work lives in `ates/`.

The immediate need is:
- A formal data model for an avalanche area (`ates/area.py`)
- A feature extraction pipeline: bbox → DEM → D8 + PRA → parquet row
- A train/test dataset built from 177 Avalanche Canada validation zones

---

## Proposed Class Design

**New file: `ates/area.py`** — uses **Pydantic** for field validation and serialisation.

```python
from pydantic import BaseModel, model_validator

class BoundingBox(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @model_validator(mode="after")
    def check_ordering(self) -> "BoundingBox": ...

    def buffer(self, deg: float = 0.05) -> "BoundingBox": ...
    def to_tuple(self) -> tuple[float, float, float, float]: ...


class AvalancheArea(BaseModel):
    name: str
    bbox: BoundingBox

    def download_mrdem(self) -> tuple[np.ndarray, Affine]:
        # thin wrapper around ates/dem.py:fetch_dem_mrdem — reuse, don't duplicate

    def generate_d8_pra(self, dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # fill_pits → flowdir → PRA-weighted accumulation (pysheds)
        # PRA logic refactored from autoates_v2/PRA/ into ates/pra.py (pure numpy, no file I/O)
        # returns (d8_exposure, pra_binary)


class AreaDataset(BaseModel):
    areas: list[AvalancheArea]
    out_dir: Path

    def build(self) -> pd.DataFrame:
        # for each area: download_mrdem → generate_d8_pra → features.extract
        # → flatten to pixel rows → checkpoint area.parquet → return concatenated DataFrame

    def spatial_split(self, n_splits: int = 5) -> Generator:
        # StratifiedGroupKFold(groups=df['area'])
        # yields (train_df, test_df) — each fold holds out one complete area
```

Pydantic gives us: bbox coordinate ordering validation, JSON serialisation (area configs
can be stored as JSON), and `model_validate()` for loading from existing metadata dicts.

---

## System Architecture — Zoomed Out

There are two distinct flows that need to be separated clearly:

### Flow A — Offline Training (runs once, or on demand)
```
177 KMZ zones (Avalanche Canada)
    → Step Functions Map state
        → Lambda: AvalancheArea.download_mrdem + generate_d8_pra + features.extract
        → writes area.parquet to S3
    → Step Functions aggregator step
        → concatenates → s3://bucket/dataset/pixels.parquet
→ Training script (SageMaker / local notebook)
    → fits RandomForest / HistGradientBoosting
    → writes model.pkl → s3://bucket/models/autoatesv3.pkl
```

**Trigger:** Manual (developer runs CDK deploy + start-execution), or a scheduled
EventBridge rule (e.g. nightly rebuild when new KMZ zones are added).
Retraining does **not** happen inside Lambda — the model is a static artefact on S3.

### Flow B — Online Inference (per user request)
```
User submits bbox (Streamlit / API Gateway)
    → Lambda: AvalancheArea(bbox).download_mrdem + generate_d8_pra + features.extract
    → loads model.pkl from S3 (cached in /tmp after first call)
    → returns ATES classification raster (GeoTIFF bytes or JSON grid)
→ Streamlit renders result on map
```

**Trigger:** Direct Lambda invoke from the Streamlit app, or API Gateway POST.
The same `AvalancheArea` methods power both flows — the class is the shared contract.

---

## Train/Test Split Standards — Spatial ML

**Core problem:** spatial autocorrelation means nearby pixels are correlated — a random split
leaks training signal into the test set and gives inflated accuracy.

**Fix:** `StratifiedGroupKFold(n_splits=5, groups=df['area'])` — each fold holds out a
complete geographic area, so the model is always tested on terrain it has never seen.

**Key references:**

1. **Roberts et al. 2017** — "Cross-validation strategies for data with temporal, spatial,
   hierarchical, or phylogenetic structure", *Ecography*. The canonical paper on spatial CV.
   https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881

2. **CAST R package** (Meyer et al.) — Nearest-Neighbour Distance Matching (NNDM) for
   measuring extrapolation distance. Useful framing even if not using R.
   https://github.com/HannaMeyer/CAST

3. **EO-learn** (Sinergise / ESA) — scikit-learn compatible framework with `EOPatch` as the
   unit of work (directly analogous to `AvalancheArea`). Good pipeline structure reference.
   https://eo-learn.readthedocs.io

4. **GeoParquet** — the format for `pixels.parquet`. GeoPandas writes it natively.
   https://geoparquet.org

5. **STAC** — standard for cataloguing geospatial datasets (MRDEM itself is STAC-compatible).
   https://stacspec.org

---

## Critical Files

| File | Role |
|------|------|
| `ates/area.py` | **New** — BoundingBox + AvalancheArea + AreaDataset (Pydantic) |
| `ates/pra.py` | **New** — PRA logic refactored from `autoates_v2/PRA/PRA_AutoATES.py` into pure numpy |
| `ates/dem.py` | Existing — `fetch_dem_mrdem` wrapped by `AvalancheArea.download_mrdem` |
| `ates/areas.py` | Existing — area metadata dicts; `AvalancheArea.model_validate()` should accept same shape |
| `ates/validate.py` | Existing — `Validator._load_zones()` reused by `AreaDataset` to load KMZ zones |
| `ates/features.py` | **New** (Week 3) — slope, aspect, TRI, TPI, curvature, D8, forest, PRA extraction |
| `scripts/build_dataset.py` | Existing — orchestration; will call `AreaDataset.build()` |
| `infra/` | **New** (Week 5) — CDK stack: S3 + Lambda + Step Functions |

---

## Verification

1. `BoundingBox(min_lat=52, min_lon=-117, max_lat=51, max_lon=-116)` raises a Pydantic `ValidationError` (lat ordering check).
2. `AvalancheArea("bow-summit", bbox).download_mrdem()` returns an array with shape > (10, 10) and a valid Affine.
3. `generate_d8_pra(dem)` returns two arrays matching `dem.shape`, both ≥ 0.
4. Full pipeline on Bow Summit bbox → `d8_exposure` compared against `cell_counts.tif` via the confusion matrix in `d8-flow.ipynb`.
5. `AreaDataset.build()` on 3 areas → `pixels.parquet` schema matches plan schema with `area` column present.
****