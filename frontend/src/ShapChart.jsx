// Diverging horizontal bar chart of SHAP contributions — no charting dependency.

const PRETTY = {
  minimum_nights_clipped: 'Min. nights',
  neighbourhood_encoded: 'Neighbourhood',
  distance_to_center_km: 'Distance to centre',
  is_shared_bath: 'Shared bath',
  availability_ratio: 'Availability',
  review_velocity: 'Review velocity',
  is_multi_listing_host: 'Multi-listing host',
  host_is_superhost: 'Superhost',
  host_listings_count: 'Host listings',
  amenity_count: 'Amenity count',
  number_of_reviews: 'Number of reviews',
  reviews_per_month: 'Reviews / month',
  review_scores_rating: 'Rating',
  review_scores_cleanliness: 'Cleanliness score',
  review_scores_location: 'Location score',
  review_scores_value: 'Value score',
}

function prettyName(feature) {
  if (PRETTY[feature]) return PRETTY[feature]
  if (feature.startsWith('has_')) return feature.slice(4).replace(/_/g, ' ')
  if (feature.startsWith('room_')) return feature.slice(5)
  if (feature.startsWith('prop_')) return feature.slice(5)
  return feature.replace(/_/g, ' ')
}

export default function ShapChart({ contributions }) {
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 1e-6)

  return (
    <div className="shap">
      {contributions.map((c) => {
        const pct = (Math.abs(c.contribution) / maxAbs) * 50 // half-width max
        const positive = c.contribution >= 0
        return (
          <div className="shap-row" key={c.feature}>
            <span className="shap-name" title={c.feature}>{prettyName(c.feature)}</span>
            <div className="shap-track">
              <span className="shap-mid" />
              <span
                className={`shap-bar ${positive ? 'pos' : 'neg'}`}
                style={{
                  width: `${pct}%`,
                  left: positive ? '50%' : `${50 - pct}%`,
                }}
              />
            </div>
            <span className={`shap-val ${positive ? 'pos' : 'neg'}`}>
              {positive ? '+' : '−'}{Math.abs(c.contribution).toFixed(2)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
