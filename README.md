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

## ATES V2 Technical Model

https://avalanchejournal.ca/the-avalanche-terrain-exposure-scale-v-2/

| Factor | Class 0: Non-Avalanche Terrain | Class 1: Simple Terrain | Class 2: Challenging Terrain | Class 3: Complex Terrain | Class 4: Extreme Terrain |
|---|---|---|---|---|---|
| **Exposure** | No known exposure to avalanche paths | Minimal exposure crossing low-frequency runout zones or short slopes only | Intermittent exposure managing a single path or paths with separation | **Frequent exposure to starting zones, tracks or multiple overlapping paths** | Sustained exposure within or immediately below starting zones |
| **Slope angle and Forest density** | Very low angle (< 15°) open terrain unconnected to steeper slopes, or steeper areas in dense forest | Low angle (15°-25°) open terrain with isolated small (< Size 2) moderate-angle slopes and/or forest openings for runout zones | Moderate-angle (25°-35°) open terrain with isolated large (≤ Size 3) high angle slopes in glades or open areas | Large proportion of high-angle (35°-45°) open or gladed terrain, but mostly moderate-angle terrain | Large proportion of very-high angle (> 45°) terrain with few or no trees |
| **Slope shape** | Straightforward, flat or undulating terrain | Straightforward undulating terrain | Mostly undulating with isolated slopes of planar, convex or concave shape | Convoluted with multiple open slopes of intricate and varied terrain shapes | Intricate, often cliffy terrain with couloirs, spines and/or overhung by cornices |
| **Terrain traps** | No avalanche related terrain traps | Occasional creek beds, tree wells or drop-offs | Single slopes above gullies or risk of impact into trees or rocks | Multiple slopes above gullies and/or risk of impact into trees, rocks or crevasses | Steep faces with cliffs, cornices, crevasses and/or risk of impact into trees or rocks |
| **Frequency-magnitude** (avalanches:years) | Never > Size 1 | **< 1:100 - 1:30 for ≥ Size 2** | 1:1 for < Size 2; **1:30 - 1:3 for ≥ Size 2** | 1:1 for Size 3; **1:1 for ≥ Size 3** | 10:1 for ≤ Size 2; > 1:1 for > Size 2 |
| **Starting zone size and density** | No known starting zones | Runout zones only except for isolated, small starting zones with Size 2 potential | Isolated starting zones with ≤ Size 3 potential or several start zones with ≤ Size 2 potential | Multiple starting zones capable of producing avalanches of all sizes | Many very large starting zones capable of producing avalanches of all sizes |
| **Runout zone characteristics** | No known runout zones | Clear boundaries, gentle transitions, smooth runouts, no connection to starting zones above | Abrupt transitions, confined runouts, long connection to starting zones above | Multiple converging paths, confined runouts, connected to starting zones above | Steep fans, confined gullies, cliffs, crevasses, starting zones directly overhead |
| **Route options** | Designated trails or low-angle areas with many options | Numerous, terrain allows multiple choices; route often obvious | **A selection of choices of varying exposure; options exist to avoid avalanche paths** | Limited options to reduce exposure; avoidance not possible | No options to reduce exposure |

*Bold italicised text indicates default values that automatically place the ATES rating in that category or higher. Class 0 is optional due to the reliability needed to make this assessment; otherwise, Class 1 includes Class 0 terrain.*

---

## References

- CAA (2016). *Avalanche Terrain Exposure Scale: Implementation Guidelines.*
- Statham et al. (2006). *ATES — Avalanche Terrain Exposure Scale.* ISSW.


