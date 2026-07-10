"""
FastAPI backend for the Airbnb pricing model.

Endpoints:
  GET  /api/options  → dropdown choices + defaults for the frontend form
  POST /api/predict  → { price, base_price, contributions[] } for a listing
  GET  /api/health   → liveness check

Run (from project root):
  uvicorn api.main:app --reload --port 8000
"""
import os
import sys

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
        "neigh_centroids": META["neigh_centroids"],   # {name: [lat, lng]}
        "amenity_labels": META["amenity_labels"],   # {col: display name}
        "amenity_count_median": META["amenity_count_median"],
        "median_price": META["median_price"],
    }


@app.get("/api/map")
def map_points():
    """Sampled listing coordinates + price for the price map, plus the colour range."""
    return {"points": MAP_POINTS, "price_lo": MAP_PRICE_LO, "price_hi": MAP_PRICE_HI}


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
