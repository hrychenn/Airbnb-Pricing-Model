"""
Airbnb Dynamic Pricing Tool — Streamlit app.

Loads the tuned XGBoost artifact (models/artifacts/xgb_final.joblib) plus the
app metadata lookup, turns a handful of host inputs into the exact 51-feature
vector the model expects, and returns a recommended nightly price with a SHAP
waterfall explaining the prediction.
"""
import os

import joblib
import numpy as np
import pandas as pd

# Resolve artifact paths relative to this file so the app runs from any cwd
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, '..', 'models', 'artifacts', 'xgb_final.joblib')
META_PATH = os.path.join(HERE, '..', 'models', 'artifacts', 'app_metadata.joblib')


def load_artifacts(model_path=MODEL_PATH, meta_path=META_PATH):
    return joblib.load(model_path), joblib.load(meta_path)


def build_feature_row(inputs: dict, feature_cols: list, meta: dict) -> pd.DataFrame:
    """Turn the UI inputs into a single-row DataFrame with exactly `feature_cols`."""
    row = {c: 0.0 for c in feature_cols}

    # --- direct numeric fields ---
    row['accommodates'] = inputs['accommodates']
    row['bedrooms'] = inputs['bedrooms']
    row['beds'] = inputs['beds']
    row['bathrooms'] = inputs['bathrooms']
    row['is_shared_bath'] = int(inputs['is_shared_bath'])
    row['number_of_reviews'] = inputs['number_of_reviews']
    row['reviews_per_month'] = inputs['reviews_per_month']
    row['host_is_superhost'] = int(inputs['host_is_superhost'])
    row['host_listings_count'] = inputs['host_listings_count']

    # --- review sub-scores: driven by one overall rating ---
    for c in ('review_scores_rating', 'review_scores_cleanliness',
              'review_scores_location', 'review_scores_value'):
        row[c] = inputs['review_score']

    # --- derived fields (mirror notebook 02) ---
    row['minimum_nights_clipped'] = min(inputs['minimum_nights'], 30)
    row['availability_ratio'] = inputs['availability_365'] / 365
    row['review_velocity'] = inputs['number_of_reviews'] / max(inputs['host_listings_count'], 1)
    row['is_multi_listing_host'] = int(inputs['host_listings_count'] > 1)
    row['amenity_count'] = inputs['amenity_count']

    # --- location: look up encoding + distance from metadata ---
    nb = inputs['neighbourhood']
    row['neighbourhood_encoded'] = meta['neigh_encoded'].get(nb, meta['global_encoded'])
    row['distance_to_center_km'] = meta['neigh_distance'].get(nb, meta['global_distance'])

    # --- amenity flags ---
    for flag in inputs['amenities']:          # list of has_* column names
        if flag in row:
            row[flag] = 1

    # --- one-hot: room type & property type (reference categories have no column) ---
    room_col = f"room_{inputs['room_type']}"
    if room_col in row:
        row[room_col] = 1
    prop_col = f"prop_{inputs['property_type']}"
    if prop_col in row:
        row[prop_col] = 1

    return pd.DataFrame([row])[feature_cols]


def predict_price(inputs, artifact, meta):
    X = build_feature_row(inputs, artifact['feature_cols'], meta)
    log_pred = artifact['model'].predict(X)[0]
    return float(np.expm1(log_pred)), X


# ---------------------------------------------------------------------------
# Streamlit UI (only runs when launched via `streamlit run`)
# ---------------------------------------------------------------------------
def main():
    import streamlit as st
    import shap
    import matplotlib.pyplot as plt

    st.set_page_config(page_title='Airbnb Price Predictor', layout='centered')
    st.title('Airbnb Dynamic Pricing Tool')
    st.caption('ML-powered nightly price recommendations for NYC listings · tuned XGBoost, R² ≈ 0.69')

    artifact, meta = load_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        room_type = st.selectbox('Room type', meta['room_types'])
        property_type = st.selectbox('Property type', meta['property_types'])
        neighbourhood = st.selectbox('Neighbourhood', meta['neighbourhoods'])
        accommodates = st.slider('Accommodates', 1, 16, 2)
        bedrooms = st.slider('Bedrooms', 0, 10, 1)
    with col2:
        beds = st.slider('Beds', 1, 18, 1)
        bathrooms = st.number_input('Bathrooms', 0.0, 8.0, 1.0, step=0.5)
        is_shared_bath = st.checkbox('Shared bathroom')
        minimum_nights = st.number_input('Minimum nights', 1, 365, 2)
        availability_365 = st.slider('Availability (days/year)', 0, 365, 180)

    st.subheader('Reviews & host')
    c3, c4 = st.columns(2)
    with c3:
        review_score = st.slider('Review score (0–5)', 0.0, 5.0, 4.7, step=0.1)
        number_of_reviews = st.number_input('Number of reviews', 0, 2000, 10)
    with c4:
        reviews_per_month = st.number_input('Reviews per month', 0.0, 50.0, 1.0, step=0.1)
        host_listings_count = st.number_input('Host listings count', 1, 500, 1)
    host_is_superhost = st.checkbox('Superhost')

    st.subheader('Amenities')
    amenity_labels = meta['amenity_labels']
    label_to_col = {v: k for k, v in amenity_labels.items()}
    picked = st.multiselect('Select amenities', list(amenity_labels.values()),
                            default=['Wifi', 'Kitchen', 'Air conditioning'])
    amenity_count = st.slider('Approx. total number of amenities', 1, 80,
                              meta['amenity_count_median'])

    if st.button('Predict price', type='primary'):
        inputs = {
            'room_type': room_type, 'property_type': property_type,
            'neighbourhood': neighbourhood, 'accommodates': accommodates,
            'bedrooms': bedrooms, 'beds': beds, 'bathrooms': bathrooms,
            'is_shared_bath': is_shared_bath, 'minimum_nights': minimum_nights,
            'availability_365': availability_365, 'review_score': review_score,
            'number_of_reviews': number_of_reviews, 'reviews_per_month': reviews_per_month,
            'host_listings_count': host_listings_count, 'host_is_superhost': host_is_superhost,
            'amenities': [label_to_col[p] for p in picked], 'amenity_count': amenity_count,
        }
        price, X = predict_price(inputs, artifact, meta)
        st.success(f'### Recommended nightly price: ${price:,.0f}')

        st.subheader('Why this price?')
        explainer = shap.TreeExplainer(artifact['model'])
        sv = explainer(X)
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(fig, bbox_inches='tight')
        st.caption('SHAP values are in log-price space; bars show what pushed the '
                   'prediction above or below the baseline.')


if __name__ == '__main__':
    main()
