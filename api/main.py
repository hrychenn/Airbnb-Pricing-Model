"""
FastAPI backend for the Airbnb pricing model.

Endpoints:
  GET  /api/options       → dropdown choices + defaults for the frontend form
  POST /api/predict       → { price, base_price, contributions[] } for a listing
  GET  /api/map           → sampled listing coords + price for the price map
  GET  /api/distribution  → NYC price histogram + CDF (for the percentile chart)
  POST /api/whatif?vary=X → predicted-price curve as one feature is swept
  GET  /api/health        → liveness check

Run (from project root):
  uvicorn api.main:app --reload --port 8000
"""
import os
import sys

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `src` importable when running as `uvicorn api.main:app` from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.inference import load_artifacts, predict  # noqa: E402

app = FastAPI(title="Airbnb Dynamic Pricing API", version="1.0")

# Allow the Vite dev server (and any local origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup — model, metadata, and a reusable SHAP explainer
ARTIFACT, META = load_artifacts()
EXPLAINER = shap.TreeExplainer(ARTIFACT["model"])

# Sampled listing coordinates + price for the map layer (kept small for a light payload)
_clean = pd.read_csv(os.path.join(ROOT, "data", "processed", "listings_clean.csv"),
                     usecols=["latitude", "longitude", "price"])
_sample = _clean.sample(n=min(2500, len(_clean)), random_state=42)
MAP_POINTS = [
    {"lat": round(float(r.latitude), 5), "lng": round(float(r.longitude), 5),
     "price": round(float(r.price))}
    for r in _sample.itertuples()
]
# 5th/95th price percentiles for the colour scale (clamps extreme values)
MAP_PRICE_LO = float(_clean["price"].quantile(0.05))
MAP_PRICE_HI = float(_clean["price"].quantile(0.95))

# --- Price distribution (histogram for display + CDF for percentile lookup) ---
_prices = _clean["price"].values
_DIST_MAX = 1000  # histogram display cap; the long tail beyond is lumped into the last bin
_counts, _edges = np.histogram(np.clip(_prices, 30, _DIST_MAX), bins=40, range=(30, _DIST_MAX))
DIST_BINS = [{"x0": round(float(_edges[i])), "x1": round(float(_edges[i + 1])),
              "count": int(_counts[i])} for i in range(len(_counts))]
_qs = np.linspace(0, 1, 101)
DIST_CDF = [[round(float(p), 2), round(float(q), 4)]
            for p, q in zip(np.quantile(_prices, _qs), _qs)]
DIST_MEDIAN = float(np.median(_prices))

# --- What-if: which features can be swept, and over what range ---
WHATIF_RANGES = {
    "accommodates": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
    "bedrooms": [0, 1, 2, 3, 4, 5, 6],
    "bathrooms": [1, 1.5, 2, 2.5, 3, 3.5, 4],
    "minimum_nights": [1, 2, 3, 5, 7, 14, 21, 30],
    "review_score": [3.0, 3.5, 4.0, 4.3, 4.5, 4.7, 4.9, 5.0],
    "amenity_count": [5, 10, 20, 30, 40, 50, 60, 70],
}
WHATIF_LABELS = {
    "accommodates": "Accommodates", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
    "minimum_nights": "Minimum nights", "review_score": "Review score",
    "amenity_count": "Amenity count",
}


class Listing(BaseModel):
    room_type: str
    property_type: str
    neighbourhood: str
    accommodates: int = Field(ge=1, le=16)
    bedrooms: int = Field(ge=0, le=10)
    beds: int = Field(ge=1, le=18)
    bathrooms: float = Field(ge=0, le=8)
    is_shared_bath: bool = False
    minimum_nights: int = Field(ge=1, le=365)
    availability_365: int = Field(ge=0, le=365)
    review_score: float = Field(ge=0, le=5)
    number_of_reviews: int = Field(ge=0, le=2000)
    reviews_per_month: float = Field(ge=0, le=50)
    host_listings_count: int = Field(ge=1, le=500)
    host_is_superhost: bool = False
    amenities: list[str] = []
    amenity_count: int = Field(ge=1, le=80)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/options")
def options():
    """Everything the frontend needs to render the form."""
    return {
        "room_types": META["room_types"],
        "property_types": META["property_types"],
        "neighbourhoods": META["neighbourhoods"],
        "neighbourhoods_by_borough": META["neighbourhoods_by_borough"],  # {borough: [names]}
        "neigh_centroids": META["neigh_centroids"],   # {name: [lat, lng]}
        "amenity_labels": META["amenity_labels"],   # {col: display name}
        "amenity_count_median": META["amenity_count_median"],
        "median_price": META["median_price"],
        "whatif_features": [{"key": k, "label": WHATIF_LABELS[k]} for k in WHATIF_RANGES],
    }


@app.get("/api/map")
def map_points():
    """Sampled listing coordinates + price for the price map, plus the colour range."""
    return {"points": MAP_POINTS, "price_lo": MAP_PRICE_LO, "price_hi": MAP_PRICE_HI}


@app.get("/api/distribution")
def distribution():
    """NYC price histogram (for the bars) + CDF (for exact percentile lookup)."""
    return {"bins": DIST_BINS, "cdf": DIST_CDF, "median": DIST_MEDIAN, "max": _DIST_MAX}


@app.post("/api/whatif")
def whatif(listing: Listing, vary: str):
    """Sweep one feature over its range, holding the rest fixed, and return the
    predicted-price curve."""
    if vary not in WHATIF_RANGES:
        return {"error": f"unknown feature '{vary}'"}
    base = listing.model_dump()
    points = []
    for v in WHATIF_RANGES[vary]:
        inp = {**base, vary: v}
        price, _, _ = predict(inp, ARTIFACT, META)  # no SHAP needed here
        points.append({"x": v, "price": round(price, 2)})
    return {"feature": vary, "label": WHATIF_LABELS[vary],
            "current": base.get(vary), "points": points}


@app.post("/api/predict")
def predict_price(listing: Listing):
    price, contributions, base_price = predict(
        listing.model_dump(), ARTIFACT, META, explainer=EXPLAINER, top_k=10
    )
    return {
        "price": round(price, 2),
        "base_price": round(base_price, 2) if base_price is not None else None,
        "contributions": contributions,
    }
