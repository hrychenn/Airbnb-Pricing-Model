# Airbnb Dynamic Pricing Model

### 🔗 [**Live demo →**](https://airbnb-dynamic-pricing-g7d0.onrender.com/)
*Hosted free on Render — the first load after idle may take ~50 s to wake the server.*

A machine-learning tool that recommends the optimal nightly price for an NYC Airbnb listing. A host enters their listing's attributes and gets a data-driven price, a SHAP breakdown of what drives it, and interactive "what-if" curves showing how changes (more guests, another bathroom, superhost status) would move the price.

**Who it's for:** Airbnb **hosts/owners** deciding what to charge. The inputs (host listings count, availability, minimum-nights policy, superhost status) are things a host knows and controls, and the output is a single *recommended price* rather than a "fair range." The same model could power a guest-facing "is this fairly priced?" tool by reframing the output around the prediction's error band, but that isn't the goal here.

**Dataset:** NYC listings from Inside Airbnb (~30k raw → ~21k after cleaning)  
**Final model:** tuned XGBoost on 51 tabular features: 24.6% MAPE, R² 0.69 on a held-out test set  
**App:** FastAPI backend + React/Vite frontend (Streamlit fallback), containerized with Docker and [deployed live on Render](https://airbnb-dynamic-pricing-g7d0.onrender.com/)

---

## Project Structure

```
├── data/
│   ├── raw/                     # Original listings.csv (gitignored)
│   ├── processed/               # Cleaned + feature-engineered datasets (gitignored)
│   └── external/                # NYC geo data, etc.
├── notebooks/
│   ├── 01_eda.ipynb             # EDA + cleaning → listings_clean.csv
│   ├── 02_feature_engineering.ipynb   # features → features.csv
│   └── 03_modeling.ipynb        # model comparison, tuning, SHAP → xgb_final.joblib
├── src/
│   ├── inference.py             # shared feature-building + predict (API & Streamlit)
│   ├── build_app_metadata.py    # builds app_metadata.joblib (dropdowns, centroids)
│   ├── features/                # feature-engineering helpers
│   ├── models/                  # training/tuning/eval helpers
│   └── visualization/           # reusable plot helpers
├── api/
│   └── main.py                  # FastAPI backend (predict, map, distribution, whatif)
├── frontend/                    # React + Vite SPA (the dashboard)
│   ├── index.html
│   ├── vite.config.js           # dev server + /api proxy → :8000
│   ├── package.json
│   └── src/
│       ├── App.jsx              # form + result layout
│       ├── ShapChart.jsx        # per-prediction SHAP contribution bars
│       ├── Distribution.jsx     # NYC price histogram + percentile
│       ├── WhatIf.jsx           # interactive price-sensitivity curves
│       ├── PriceMap.jsx         # Leaflet NYC price map
│       ├── api.js               # fetch wrappers for the backend
│       └── index.css            # styling
├── app/
│   └── app.py                   # Streamlit fallback UI
├── models/artifacts/            # xgb_final.joblib + app_metadata.joblib (gitignored)
├── reports/figures/             # saved charts from the notebooks
├── .claude/launch.json          # dev-server launch configs
├── requirements.txt
└── tests/
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download NYC listings from [Inside Airbnb](http://insideairbnb.com/get-the-data/) and place `listings.csv` in `data/raw/`.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Price distributions, neighbourhood analysis, outlier handling |
| `02_feature_engineering.ipynb` | Amenity parsing, embeddings, target encoding, log-transform |
| `03_modeling.ipynb` | Model comparison, Optuna tuning, SHAP explanations |

---

## Results

Held-out test set (20%, ~4,250 listings). Metrics computed on dollar prices after inverting the log transform.

| Model | MAPE | RMSE | R² |
|---|---|---|---|
| Naive baseline (mean price) | 107.8% | $243 | ~0.00 |
| Ridge regression | 33.2% | $170 | 0.51 |
| Random forest | 26.9% | $146 | 0.64 |
| XGBoost (tabular) | 25.2% | $137 | 0.68 |
| XGBoost + HF embeddings | 25.3% | $138 | 0.68 |
| **XGBoost (tuned)** — deployed | **24.6%** | **$135** | **0.69** |

**Notes on results:**
- The HuggingFace amenity embeddings did **not** improve on the hand-engineered binary flags (an ablation in notebook 03 shows they are *substitutes*, not complements). The simpler tabular model is deployed.
- Final model clears the revised target (MAPE < 25%, R² > 0.65). The original 20% MAPE target proved optimistic for a single-scrape, tabular-only model. The remaining error is concentrated in the luxury tail, which needs listing photos/description text to model. See the "Success Criteria" section in notebook 03.

---

## Deployment

The app is a **FastAPI backend** (serves the model) + a **React/Vite frontend** (the UI). A single-file Streamlit version is also kept as a fallback.

One-time, after training — build the metadata lookup the app needs:

```bash
python -m src.build_app_metadata      # writes models/artifacts/app_metadata.joblib
```

**Run the React app (two terminals):**

```bash
# terminal 1 — API on :8000
uvicorn api.main:app --reload --port 8000

# terminal 2 — UI on :5173 (proxies /api → :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. The frontend collects listing details, POSTs to `/api/predict`, and renders the recommended price plus a SHAP contribution chart explaining it.

**Fallback — Streamlit (single command):**

```bash
streamlit run app/app.py
```

### Live deployment (Render, free)

The app is containerized as a **single Docker image** — FastAPI serves both the API and the
built React app on one port. Everything the runtime needs (dropdowns, price-map sample,
distribution) is baked into `app_metadata.joblib`, and SHAP values come from XGBoost's native
`pred_contribs` (no `shap` lib) — so the image ships only code + the two `.joblib` artifacts.
It peaks at ~156 MB RAM, well under Render free's 512 MB.

```bash
docker build -t airbnb-pricing .
docker run --rm -p 7860:7860 airbnb-pricing   # → http://localhost:7860
```

Deployed on **Render** (free tier) via `render.yaml` — a push to `main` auto-rebuilds and
redeploys. *(HuggingFace Spaces' Docker SDK is paid now, so Render is the free path for this stack.)*

### Layout

```
api/main.py            FastAPI: /api/options, /api/predict, /api/map,
                       /api/distribution, /api/whatif, /api/health
src/inference.py       shared feature-building + predict (used by API and Streamlit)
src/build_app_metadata.py   builds the dropdown/lookup metadata + neighbourhood centroids
frontend/              React + Vite SPA
  App.jsx              form + result layout
  ShapChart.jsx        per-prediction SHAP contribution bars
  Distribution.jsx     NYC price histogram + your-listing percentile
  WhatIf.jsx           interactive sensitivity curve (price vs one feature)
  PriceMap.jsx         Leaflet price map of NYC (dots coloured by price)
  index.css            styling
app/app.py             Streamlit fallback UI
```

### Visualizations
- **Per-prediction SHAP chart** — diverging bars showing what pushed the price up/down.
- **Price distribution + percentile** — histogram of all NYC listing prices with your prediction marked and a "top X%" readout.
- **What-if sensitivity curves** — interactive line chart of predicted price as one feature (accommodates, bedrooms, review score, …) sweeps its range, holding the rest fixed.
- **NYC price map** — Leaflet map (free CARTO tiles, no API key) with ~2,500 listings coloured by nightly price; the selected neighbourhood is ringed and the map recenters on it.
