# Airbnb Dynamic Pricing Model

A machine learning pipeline that predicts optimal nightly prices for Airbnb listings using tabular features and HuggingFace sentence embeddings.

**Dataset:** NYC listings from Inside Airbnb (~48,000 records)  
**Final model:** XGBoost + sentence-transformer embeddings (PCA-reduced)  
**Deployment:** Streamlit app on HuggingFace Spaces

---

## Project Structure

```
├── data/
│   ├── raw/            # Original, unmodified listings CSV
│   ├── processed/      # Cleaned and feature-engineered datasets
│   └── external/       # NYC geo data, etc.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features/       # Feature engineering logic
│   ├── models/         # Training, tuning, evaluation
│   └── visualization/  # Reusable plot helpers
├── app/                # Streamlit application
├── models/artifacts/   # Serialized pipelines (joblib)
├── reports/figures/    # Saved charts for writeup
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
- Final model clears the revised target (MAPE < 25%, R² > 0.65). The original 20% MAPE target proved optimistic for a single-scrape, tabular-only model — remaining error is concentrated in the luxury tail, which needs listing photos/description text to model. See the "Success Criteria" section in notebook 03.

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
