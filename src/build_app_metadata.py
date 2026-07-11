"""
Build the lookup metadata the Streamlit app needs to turn a handful of human inputs
into the exact 51-feature vector the model expects.

Reads the trained artifact (for the train-only neighbourhood encoding) plus the
processed data (for per-neighbourhood distance and sensible defaults), and writes
models/artifacts/app_metadata.joblib.

Run:  python -m src.build_app_metadata   (from project root)

Everything the deployed app needs at runtime (dropdowns, centroids, the price-map
sample, and the price distribution) is baked into app_metadata.joblib here, so the
container only ships the two .joblib artifacts — no data CSVs required.
"""
import joblib
import numpy as np
import pandas as pd

ARTIFACT = 'models/artifacts/xgb_final.joblib'
FEATURES = 'data/processed/features.csv'
CLEAN    = 'data/processed/listings_clean.csv'
OUT      = 'models/artifacts/app_metadata.joblib'

# Human-readable labels for the 18 amenity flags
AMENITY_LABELS = {
    'has_wifi': 'Wifi',
    'has_kitchen': 'Kitchen',
    'has_air_conditioning': 'Air conditioning',
    'has_heating': 'Heating',
    'has_tv': 'TV',
    'has_washer': 'Washer',
    'has_dryer': 'Dryer',
    'has_dishwasher': 'Dishwasher',
    'has_dedicated_workspace': 'Dedicated workspace',
    'has_self_checkin': 'Self check-in',
    'has_elevator': 'Elevator',
    'has_pool': 'Pool',
    'has_hot_tub': 'Hot tub',
    'has_gym': 'Gym',
    'has_free_parking': 'Free parking',
    'has_bathtub': 'Bathtub',
    'has_long_term_stays': 'Long-term stays allowed',
    'has_first_aid_kit': 'First aid kit',
}


def main():
    artifact = joblib.load(ARTIFACT)
    feature_cols = artifact['feature_cols']
    neigh_mean = artifact['neigh_mean_train']          # Series: neighbourhood -> encoded value
    global_encoded = float(artifact['global_mean_train'])

    feats = pd.read_csv(FEATURES)
    clean = pd.read_csv(CLEAN)

    # Per-neighbourhood median distance to centre (from engineered features)
    neigh_distance = feats.groupby('neighbourhood_cleansed')['distance_to_center_km'].median()
    global_distance = float(feats['distance_to_center_km'].median())

    # Per-neighbourhood centroid (median lat/lng) so the map can mark the selected area
    centroids = clean.groupby('neighbourhood_cleansed')[['latitude', 'longitude']].median()
    neigh_centroids = {n: [float(r['latitude']), float(r['longitude'])]
                       for n, r in centroids.iterrows()}

    # Category lists (mirror notebook 02's grouping so the dummy columns line up)
    room_types = clean['room_type'].value_counts().index.tolist()
    top_props = clean['property_type'].value_counts().head(10).index.tolist()
    property_types = top_props + ['Other']

    # Neighbourhoods grouped by borough for a friendlier dropdown. Only offer
    # neighbourhoods the model actually learned an encoding for (neigh_mean index).
    known = set(neigh_mean.index)
    borough_order = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    grouped = (clean[clean['neighbourhood_cleansed'].isin(known)]
               .groupby('neighbourhood_group_cleansed')['neighbourhood_cleansed']
               .apply(lambda s: sorted(s.unique())).to_dict())
    neighbourhoods_by_borough = {b: grouped[b] for b in borough_order if b in grouped}
    for b, names in grouped.items():          # append any borough not in the preset order
        neighbourhoods_by_borough.setdefault(b, names)

    # --- Price-map sample (bake in so the runtime doesn't need the raw CSV) ---
    sample = clean.sample(n=min(2500, len(clean)), random_state=42)
    map_points = [{"lat": round(float(r.latitude), 5), "lng": round(float(r.longitude), 5),
                   "price": round(float(r.price))} for r in sample.itertuples()]
    map_price_lo = float(clean["price"].quantile(0.05))
    map_price_hi = float(clean["price"].quantile(0.95))

    # --- Price distribution: histogram (display) + CDF (percentile lookup) ---
    prices = clean["price"].values
    dist_max = 1000  # display cap; the long tail beyond is lumped into the last bin
    counts, edges = np.histogram(np.clip(prices, 30, dist_max), bins=40, range=(30, dist_max))
    dist_bins = [{"x0": round(float(edges[i])), "x1": round(float(edges[i + 1])),
                  "count": int(counts[i])} for i in range(len(counts))]
    qs = np.linspace(0, 1, 101)
    dist_cdf = [[round(float(p), 2), round(float(q), 4)]
                for p, q in zip(np.quantile(prices, qs), qs)]
    dist_median = float(np.median(prices))

    metadata = {
        'feature_cols': feature_cols,
        'neighbourhoods': sorted(neigh_mean.index.tolist()),
        'neigh_encoded': neigh_mean.to_dict(),
        'neigh_distance': neigh_distance.to_dict(),
        'global_encoded': global_encoded,
        'global_distance': global_distance,
        'room_types': room_types,
        'property_types': property_types,
        'neigh_centroids': neigh_centroids,
        'neighbourhoods_by_borough': neighbourhoods_by_borough,
        'amenity_labels': AMENITY_LABELS,
        # Defaults for fields the UI doesn't expose directly
        'amenity_count_median': int(feats['amenity_count'].median()),
        'median_price': float(feats['price'].median()),
        # Baked-in viz data so the runtime needs no data CSV
        'map_points': map_points,
        'map_price_lo': map_price_lo,
        'map_price_hi': map_price_hi,
        'dist_bins': dist_bins,
        'dist_cdf': dist_cdf,
        'dist_median': dist_median,
        'dist_max': dist_max,
    }

    joblib.dump(metadata, OUT)
    print(f'Wrote {OUT}')
    print(f'  neighbourhoods: {len(metadata["neighbourhoods"])}')
    print(f'  boroughs: {[(b, len(n)) for b, n in neighbourhoods_by_borough.items()]}')
    print(f'  room_types: {room_types}')
    print(f'  property_types: {property_types}')
    print(f'  amenity_count_median: {metadata["amenity_count_median"]}')


if __name__ == '__main__':
    main()
