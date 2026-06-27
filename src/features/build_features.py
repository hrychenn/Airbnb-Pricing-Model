"""
Feature engineering pipeline: amenity parsing, embeddings, derived features, encoding.
"""
import ast
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


COMMON_AMENITIES = [
    "wifi", "tv", "kitchen", "washer", "dryer", "air_conditioning",
    "heating", "pool", "gym", "elevator", "parking", "hot_tub",
    "dishwasher", "smoke_alarm", "carbon_monoxide_alarm",
]

NYC_CENTER = (40.7580, -73.9855)  # Times Square


def parse_amenities(amenity_str: str) -> set:
    try:
        items = ast.literal_eval(amenity_str)
        return {re.sub(r"[^a-z0-9]", "_", i.lower().strip()) for i in items}
    except Exception:
        return set()


def add_amenity_flags(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["amenities"].apply(parse_amenities)
    for amenity in COMMON_AMENITIES:
        col = f"has_{amenity}"
        df[col] = parsed.apply(lambda s: int(amenity in s))
    df["amenity_count"] = parsed.apply(len)
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    lat_r = np.radians(df["latitude"])
    lon_r = np.radians(df["longitude"])
    clat = np.radians(NYC_CENTER[0])
    clon = np.radians(NYC_CENTER[1])
    dlat = lat_r - clat
    dlon = lon_r - clon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(clat) * np.sin(dlon / 2) ** 2
    df["distance_to_center_km"] = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["availability_ratio"] = df["availability_365"] / 365
    df["review_velocity"] = df["number_of_reviews"] / (df["host_listings_count"].clip(lower=1))
    df["price_log"] = np.log1p(df["price"])
    return df


class PriceLogTransformer(BaseEstimator, TransformerMixin):
    """Log-transform target; inverse_transform gives back dollar prices."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)
