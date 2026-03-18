# ATES Rule-Based Classifier

A deterministic terrain classifier that assigns an [ATES](https://www.avalanche.ca/mountain-information/ates) rating to each point along a GPX route using published CAA decision criteria. This is Phase 1 of a broader ML pipeline — its job is to validate feature engineering and produce a reproducible baseline before any model training.

---

## How It Works

Each point along the GPX track is evaluated against five terrain features derived from a DEM:

| Feature | Source |
|---------|--------|
| Local slope angle | RichDEM |
| Overhead start zone area (ha above 30°/35°) | PySheds contributing area + alpha angle |
| Start zone width (m) | Upstream flow width estimate |
| Terrain trap presence | TPI depression + channel detection |
| Forest cover modifier | NDVI or NALCMS canopy mask |

These feed a transparent rule engine in `ates/rules.py` that mirrors the CAA ATES criteria. The route-level rating is the **P95 of per-point scores** — one unavoidable Complex window defines the route.

---

## Quickstart

```bash
pip install -r requirements.txt

# Fetch and align DEM tiles for your route
python scripts/prepare_rasters.py --gpx routes/my_route.gpx --outdir data/processed/

# Run classifier
python scripts/run_classifier.py \
  --gpx routes/my_route.gpx \
  --dem data/processed/dem.tif \
  --canopy data/processed/canopy.tif \
  --output output/ates.geojson \
  --heatmap output/heatmap.html
```

---

## Data

- **DEM** — [HRDEM](https://open.canada.ca/data/en/dataset/957782bf-847c-4644-a757-e383c0057995) (1m lidar, NRCan) for the Rockies; Copernicus GLO-30 as fallback
- **Canopy** — NALCMS 2020 land cover or Sentinel-2 NDVI (threshold > 0.5)
- **Avalanche paths** — CAA mapped paths from avalanche.ca (optional; improves overhead hazard estimate)

---

## Project Structure

```
ice-autoATES/
├── ingest.py       # GPX parsing
├── terrain.py      # Slope, TPI, curvature from DEM
├── overhead.py     # Upslope contributing area + alpha-angle runout
├── traps.py        # Terrain trap detection
├── canopy.py       # Forest cover modifier
├── rules.py        # Feature → ATES rating decision logic
└── output.py       # GeoJSON + Folium heatmap

scripts/
├── download_hrdem.py
├── prepare_rasters.py
└── run_classifier.py

data/validation/
└── labeled_routes.geojson   # Known-rating routes (Polar Circus etc.) for accuracy checks

notebooks/
└── validation.ipynb
```

---

## Validation

Known ratings from Avalanche Canada annotations and guidebook designations are stored in `data/validation/labeled_routes.geojson`. Run `notebooks/validation.ipynb` to see per-route accuracy and confusion matrix.

**Target before moving to ML:** ≥80% agreement with known ratings (within one class). Systematic misclassification points to a feature engineering bug, not a modelling problem.

---

## Limitations

- Overhead hazard uses a fixed α = 18° and contributing area rather than full viewshed — tends to underestimate on routes with hanging terrain above cliff bands
- No temporal component; snowpack depth and aspect-driven loading are ignored
- Canopy modifier derived from summer imagery

---

## References

- CAA (2016). *Avalanche Terrain Exposure Scale: Implementation Guidelines.*
- Statham et al. (2006). *ATES — Avalanche Terrain Exposure Scale.* ISSW.