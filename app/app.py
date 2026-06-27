"""Streamlit app: Airbnb Dynamic Pricing Tool."""
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = "../models/artifacts/xgb_pipeline.joblib"

st.set_page_config(page_title="Airbnb Price Predictor", layout="centered")
st.title("Airbnb Dynamic Pricing Tool")
st.markdown("Enter your listing details to get an ML-powered price recommendation.")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    accommodates = st.slider("Accommodates", 1, 16, 2)
    bedrooms = st.slider("Bedrooms", 0, 10, 1)
    bathrooms = st.slider("Bathrooms", 0.0, 8.0, 1.0, step=0.5)

with col2:
    neighbourhood = st.text_input("Neighbourhood", "Williamsburg")
    review_score = st.slider("Review score", 0.0, 5.0, 4.7, step=0.1)
    availability = st.slider("Availability (days/year)", 0, 365, 180)
    is_superhost = st.checkbox("Superhost")

amenities = st.text_area(
    "Amenities (comma-separated)",
    "Wifi, Kitchen, Air conditioning, Washer, Dryer",
)

if st.button("Predict Price"):
    try:
        pipeline = joblib.load(MODEL_PATH)
        row = pd.DataFrame([{
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "neighbourhood_cleansed": neighbourhood,
            "review_scores_rating": review_score,
            "availability_365": availability,
            "host_is_superhost": int(is_superhost),
            "amenities": f"[{amenities}]",
            "number_of_reviews": 10,
            "host_listings_count": 1,
            "latitude": 40.7128,
            "longitude": -74.0060,
        }])
        log_pred = pipeline.predict(row)[0]
        price = np.expm1(log_pred)
        st.success(f"**Recommended nightly price: ${price:.0f}**")

        # SHAP waterfall
        explainer = shap.Explainer(pipeline.named_steps["model"])
        X_transformed = pipeline[:-1].transform(row)
        shap_values = explainer(X_transformed)
        st.subheader("Why this price?")
        fig, ax = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Model not yet trained. Run notebook 03 to generate the pipeline artifact.")
