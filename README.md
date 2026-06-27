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

| Model | MAPE | RMSE | R² |
|---|---|---|---|
| Naive baseline | ~38% | — | — |
| Ridge regression | — | — | — |
| Random forest | — | — | — |
| XGBoost | — | — | — |
| XGBoost + HF embeddings | — | — | — |

---

## Deployment

```bash
cd app
streamlit run app.py
```
