"""
Shared inference logic: load artifacts, build the 51-feature vector from human
inputs, and predict a price with SHAP contributions. Used by both the FastAPI
backend (api/main.py) and the Streamlit app (app/app.py).
"""
import os

import joblib
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, '..', 'models', 'artifacts', 'xgb_final.joblib')
META_PATH = os.path.join(_HERE, '..', 'models', 'artifacts', 'app_metadata.joblib')


def load_artifacts(model_path=MODEL_PATH, meta_path=META_PATH):
    return joblib.load(model_path), joblib.load(meta_path)


def build_feature_row(inputs: dict, feature_cols: list, meta: dict) -> pd.DataFrame:
    """Turn UI inputs into a single-row DataFrame with exactly `feature_cols`."""
    row = {c: 0.0 for c in feature_cols}

    row['accommodates'] = inputs['accommodates']
    row['bedrooms'] = inputs['bedrooms']
    row['beds'] = inputs['beds']
    row['bathrooms'] = inputs['bathrooms']
    row['is_shared_bath'] = int(inputs['is_shared_bath'])
    row['number_of_reviews'] = inputs['number_of_reviews']
    row['reviews_per_month'] = inputs['reviews_per_month']
    row['host_is_superhost'] = int(inputs['host_is_superhost'])
    row['host_listings_count'] = inputs['host_listings_count']

    for c in ('review_scores_rating', 'review_scores_cleanliness',
              'review_scores_location', 'review_scores_value'):
        row[c] = inputs['review_score']

    row['minimum_nights_clipped'] = min(inputs['minimum_nights'], 30)
    row['availability_ratio'] = inputs['availability_365'] / 365
    row['review_velocity'] = inputs['number_of_reviews'] / max(inputs['host_listings_count'], 1)
    row['is_multi_listing_host'] = int(inputs['host_listings_count'] > 1)
    row['amenity_count'] = inputs['amenity_count']

    nb = inputs['neighbourhood']
    row['neighbourhood_encoded'] = meta['neigh_encoded'].get(nb, meta['global_encoded'])
    row['distance_to_center_km'] = meta['neigh_distance'].get(nb, meta['global_distance'])

    for flag in inputs['amenities']:
        if flag in row:
            row[flag] = 1

    room_col = f"room_{inputs['room_type']}"
    if room_col in row:
        row[room_col] = 1
    prop_col = f"prop_{inputs['property_type']}"
    if prop_col in row:
        row[prop_col] = 1

    return pd.DataFrame([row])[feature_cols]


def predict(inputs, artifact, meta, with_contribs=False, top_k=10):
    """Return (price, shap_contributions, base_value_price).

    shap_contributions: list of {feature, value, contribution} sorted by |contribution|,
    truncated to top_k. Contributions are exact tree-SHAP values in log-price space,
    computed by XGBoost's native `pred_contribs` (what shap.TreeExplainer calls under
    the hood) — so we get identical explanations without the heavy `shap` dependency.
    """
    import xgboost as xgb

    model = artifact['model']
    X = build_feature_row(inputs, artifact['feature_cols'], meta)
    log_pred = float(model.predict(X)[0])
    price = float(np.expm1(log_pred))

    contributions = []
    base_price = None
    if with_contribs:
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        # shape (1, n_features + 1); the last column is the base/expected value
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0]
        values, base = contribs[:-1], float(contribs[-1])
        base_price = float(np.expm1(base))
        order = np.argsort(np.abs(values))[::-1][:top_k]
        for i in order:
            contributions.append({
                'feature': X.columns[i],
                'value': float(X.iloc[0, i]),
                'contribution': float(values[i]),
            })

    return price, contributions, base_price
