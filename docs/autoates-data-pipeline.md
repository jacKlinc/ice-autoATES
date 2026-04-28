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

    async def download_mrdem_async(self, executor) -> tuple[np.ndarray, Affine]:
        # runs download_mrdem in a thread pool so the event loop stays unblocked
        # rasterio is blocking I/O — it must be wrapped, not awaited directly

    def generate_d8_pra(self, dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 1. slope from np.gradient → arctan → degrees
        # 2. PRA binary: cells where 30° ≤ slope ≤ 60°
        # 3. fill_pits → D8 flowdir → accumulation weighted by PRA binary
        # returns (pra_binary, d8_exposure)


class AreaDataset(BaseModel):
    areas: list[AvalancheArea]
    out_dir: Path

    async def build(self) -> pd.DataFrame:
        # asyncio.Semaphore caps concurrent S3 requests
        # asyncio.gather fans out all downloads, then compute runs per-area
        # checkpoints each area.parquet before concatenating

    def spatial_split(self, df: pd.DataFrame, n_splits: int = 5) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        # StratifiedGroupKFold(groups=df['area'])
        # yield (train_df, test_df) one fold at a time — avoids materialising all 5 folds at once
```

Pydantic gives us: bbox coordinate ordering validation, JSON serialisation (area configs
can be stored as JSON), and `model_validate()` for loading from existing metadata dicts.

---

## Generator Pattern

`spatial_split` uses `yield` to produce one `(train_df, test_df)` fold at a time:

```python
def spatial_split(self, df: pd.DataFrame, n_splits: int = 5):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(df, df["ates_class"], groups=df["area"]):
        yield df.iloc[train_idx], df.iloc[test_idx]
```

The caller iterates naturally:
```python
for train, test in dataset.spatial_split(pixels):
    model.fit(train[features], train["ates_class"])
    evaluate(model, test)
```

**Why a generator here:** each fold is a large DataFrame slice — materialising all five
at once would triple peak memory. `yield` hands one fold to the caller, which trains and
evaluates, then garbage-collects before the next fold is produced.

---

## Async Strategy

The bottleneck in `AreaDataset.build()` is network I/O — 177 sequential MRDEM COG range
requests over HTTP. `asyncio` eliminates that wait by overlapping downloads.

**Why `run_in_executor` and not bare `await`:**
rasterio's `open()` and `read()` are blocking C calls. Calling them directly inside an
async function would freeze the event loop for every other coroutine. Wrapping them in a
`ThreadPoolExecutor` hands the blocking call to a worker thread and yields control back to
the event loop until the result is ready.

```python
async def download_mrdem_async(self, executor):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, self._download_mrdem_sync)

async def build(self) -> pd.DataFrame:
    sem = asyncio.Semaphore(8)          # cap concurrent S3 connections
    executor = ThreadPoolExecutor(max_workers=8)

    async def fetch_one(area):
        async with sem:
            return area.name, await area.download_mrdem_async(executor)

    downloads = await asyncio.gather(*[fetch_one(a) for a in self.areas])

    # compute (CPU-bound) runs synchronously after each download
    frames = []
    for name, (dem, affine) in downloads:
        pra, exposure = area.generate_d8_pra(dem)
        frames.append(features.extract(dem, pra, exposure, affine, name))
    return pd.concat(frames)
```

**Key concepts this introduces:**
| Concept | Where it appears |
|---------|-----------------|
| `async def` / `await` | `download_mrdem_async`, `build` |
| `asyncio.gather` | fan-out of 177 downloads |
| `asyncio.Semaphore` | cap concurrent S3 connections to ~8 |
| `ThreadPoolExecutor` | wrap blocking rasterio calls |
| `run_in_executor` | bridge between async and blocking code |

CPU-bound work (numpy gradient, pysheds D8) stays synchronous — `asyncio` only helps
with I/O-bound waiting. If compute becomes the bottleneck, swap `ThreadPoolExecutor` for
`ProcessPoolExecutor` to get true parallelism across cores.

---

## System Architecture — Zoomed Out

There are two distinct flows:

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

**Trigger:** Manual or scheduled EventBridge rule.
Retraining does **not** happen inside Lambda — the model is a static artefact on S3.

### Flow B — Online Inference (per user request)
```
User submits bbox (Streamlit / API Gateway)
    → Lambda: AvalancheArea(bbox).download_mrdem + generate_d8_pra + features.extract
    → loads model.pkl from S3 (cached in /tmp after first call)
    → returns ATES classification raster (GeoTIFF bytes or JSON grid)
→ Streamlit renders result on map
```

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

## Dependencies, Testing & CI/CD

Dependencies
- [ ] Remove outdated deps
- [ ] Move deps to dev

Tests
- [ ] Extend unit tests
- [ ] Add E2E pipeline tests

CI (Optional)
- [ ] Add ruff linting 
- [ ] Add test coverage dynamic SVG badges to README.md

---

## Critical Files

| File | Role |
|------|------|
| `ates/area.py` | **New** — BoundingBox + AvalancheArea + AreaDataset (Pydantic + async) |
| `ates/dem.py` | Existing — `fetch_dem_mrdem` wrapped by `AvalancheArea.download_mrdem` |
| `ates/areas.py` | Existing — area metadata dicts; `AvalancheArea.model_validate()` should accept same shape |
| `ates/validate.py` | Existing — `Validator._load_zones()` reused by `AreaDataset` to load KMZ zones |
| `ates/features.py` | **New** — slope, aspect, TRI, TPI, curvature, D8, forest, PRA extraction |
| `scripts/build_dataset.py` | Existing — orchestration; will call `asyncio.run(AreaDataset.build())` |
| `infra/` | **New** — CDK stack: S3 + Lambda + Step Functions |

---

## Verification

1. `BoundingBox(min_lat=52, min_lon=-117, max_lat=51, max_lon=-116)` raises a Pydantic `ValidationError`.
2. `AvalancheArea("bow-summit", bbox).download_mrdem()` returns an array with shape > (10, 10) and a valid Affine.
3. `generate_d8_pra(dem)` returns two arrays matching `dem.shape`, both ≥ 0.
4. Full pipeline on Bow Summit bbox → `d8_exposure` compared against `cell_counts.tif` via the confusion matrix in `d8-flow.ipynb`.
5. `asyncio.run(AreaDataset([bow_summit], out_dir).build())` on 3 areas → `pixels.parquet` schema matches plan schema with `area` column present.
6. Semaphore test: confirm no more than 8 concurrent rasterio calls fire simultaneously (log timestamps).
