import { useEffect, useState } from 'react'
import { fetchOptions, predictPrice } from './api'
import ShapChart from './ShapChart'
import PriceMap from './PriceMap'
import Distribution from './Distribution'
import WhatIf from './WhatIf'

const DEFAULTS = {
  room_type: 'Entire home/apt',
  property_type: 'Entire rental unit',
  neighbourhood: 'Williamsburg',
  accommodates: 2,
  bedrooms: 1,
  beds: 1,
  bathrooms: 1,
  is_shared_bath: false,
  minimum_nights: 2,
  availability_365: 180,
  review_score: 4.7,
  number_of_reviews: 10,
  reviews_per_month: 1,
  host_listings_count: 1,
  host_is_superhost: false,
  amenities: ['has_wifi', 'has_kitchen', 'has_air_conditioning'],
  amenity_count: 30,
}

function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="field-label">{label}{hint && <em>{hint}</em>}</span>
      {children}
    </label>
  )
}

export default function App() {
  const [options, setOptions] = useState(null)
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [submitted, setSubmitted] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchOptions()
      .then((opts) => {
        setOptions(opts)
        setForm((f) => ({ ...f, amenity_count: opts.amenity_count_median }))
      })
      .catch((e) => setError(e.message))
  }, [])

  const set = (key) => (e) => {
    const el = e.target
    const val = el.type === 'checkbox' ? el.checked
      : el.type === 'range' || el.type === 'number' ? Number(el.value)
        : el.value
    setForm((f) => ({ ...f, [key]: val }))
  }

  const toggleAmenity = (col) => {
    setForm((f) => ({
      ...f,
      amenities: f.amenities.includes(col)
        ? f.amenities.filter((a) => a !== col)
        : [...f.amenities, col],
    }))
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setLoading(true); setError(null)
    try {
      setResult(await predictPrice(form))
      setSubmitted(form)   // snapshot for the what-if panel (only updates on predict)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (error && !options) {
    return <div className="app"><div className="error-banner">Could not reach the API: {error}. Is the backend running on port 8000?</div></div>
  }
  if (!options) return <div className="app"><div className="loading">Loading…</div></div>

  const amenities = Object.entries(options.amenity_labels) // [[col, label], ...]

  return (
    <div className="app">
      <header className="header">
        <h1>Airbnb Dynamic Pricing</h1>
        <p>Data-driven nightly price recommendations for NYC listings · tuned XGBoost, R² ≈ 0.69</p>
      </header>

      <div className="layout">
        <form className="panel form" onSubmit={onSubmit}>
          <section className="card">
            <h2>Property</h2>
            <div className="grid2">
              <Field label="Room type">
                <select value={form.room_type} onChange={set('room_type')}>
                  {options.room_types.map((r) => <option key={r}>{r}</option>)}
                </select>
              </Field>
              <Field label="Property type">
                <select value={form.property_type} onChange={set('property_type')}>
                  {options.property_types.map((p) => <option key={p}>{p}</option>)}
                </select>
              </Field>
            </div>
            <div className="grid2">
              <Field label={`Accommodates: ${form.accommodates}`}>
                <input type="range" min="1" max="16" value={form.accommodates} onChange={set('accommodates')} />
              </Field>
              <Field label={`Bedrooms: ${form.bedrooms}`}>
                <input type="range" min="0" max="10" value={form.bedrooms} onChange={set('bedrooms')} />
              </Field>
              <Field label={`Beds: ${form.beds}`}>
                <input type="range" min="1" max="18" value={form.beds} onChange={set('beds')} />
              </Field>
              <Field label="Bathrooms">
                <input type="number" min="0" max="8" step="0.5" value={form.bathrooms} onChange={set('bathrooms')} />
              </Field>
            </div>
            <label className="checkbox">
              <input type="checkbox" checked={form.is_shared_bath} onChange={set('is_shared_bath')} /> Shared bathroom
            </label>
          </section>

          <section className="card">
            <h2>Location</h2>
            <Field label="Neighbourhood" hint=" (grouped by borough)">
              <select value={form.neighbourhood} onChange={set('neighbourhood')}>
                {Object.entries(options.neighbourhoods_by_borough).map(([borough, names]) => (
                  <optgroup key={borough} label={borough}>
                    {names.map((n) => <option key={n} value={n}>{n}</option>)}
                  </optgroup>
                ))}
              </select>
            </Field>
          </section>

          <section className="card">
            <h2>Booking & reviews</h2>
            <div className="grid2">
              <Field label="Minimum nights">
                <input type="number" min="1" max="365" value={form.minimum_nights} onChange={set('minimum_nights')} />
              </Field>
              <Field label={`Availability: ${form.availability_365} d/yr`}>
                <input type="range" min="0" max="365" value={form.availability_365} onChange={set('availability_365')} />
              </Field>
              <Field label={`Review score: ${form.review_score.toFixed(1)}`}>
                <input type="range" min="0" max="5" step="0.1" value={form.review_score} onChange={set('review_score')} />
              </Field>
              <Field label="Number of reviews">
                <input type="number" min="0" max="2000" value={form.number_of_reviews} onChange={set('number_of_reviews')} />
              </Field>
              <Field label="Reviews / month">
                <input type="number" min="0" max="50" step="0.1" value={form.reviews_per_month} onChange={set('reviews_per_month')} />
              </Field>
              <Field label="Host listings count">
                <input type="number" min="1" max="500" value={form.host_listings_count} onChange={set('host_listings_count')} />
              </Field>
            </div>
            <label className="checkbox">
              <input type="checkbox" checked={form.host_is_superhost} onChange={set('host_is_superhost')} /> Superhost
            </label>
          </section>

          <section className="card">
            <h2>Amenities</h2>
            <div className="chips">
              {amenities.map(([col, label]) => (
                <button
                  type="button"
                  key={col}
                  className={`chip ${form.amenities.includes(col) ? 'chip-on' : ''}`}
                  onClick={() => toggleAmenity(col)}
                >
                  {label}
                </button>
              ))}
            </div>
            <Field label={`Approx. total number of amenities: ${form.amenity_count}`} hint=" (listings average ~30)">
              <input type="range" min="1" max="80" value={form.amenity_count} onChange={set('amenity_count')} />
            </Field>
          </section>

          <button className="predict-btn" type="submit" disabled={loading}>
            {loading ? 'Predicting…' : 'Predict price'}
          </button>
        </form>

        <aside className="panel result">
          {result ? (
            <>
              <div className="price-card">
                <span className="price-label">Recommended nightly price</span>
                <span className="price-value">${Math.round(result.price).toLocaleString()}</span>
                {result.base_price != null && (
                  <span className="price-base">avg NYC listing ≈ ${Math.round(result.base_price).toLocaleString()}</span>
                )}
              </div>
              <h3>Why this price?</h3>
              <ShapChart contributions={result.contributions} />
              <p className="foot">Bars show how each feature pushes the prediction above (coral) or below (blue) the average listing. Values are in log-price space.</p>
            </>
          ) : (
            <div className="empty">
              <div className="empty-icon">🏠</div>
              <p>Fill in the listing details and hit <strong>Predict price</strong> to see a recommendation and a breakdown of what drives it.</p>
            </div>
          )}
          {error && <div className="error-banner">{error}</div>}
        </aside>
      </div>

      {result && submitted && (
        <>
          <section className="card viz-card">
            <h2>Market context</h2>
            <Distribution price={result.price} />
          </section>
          <section className="card viz-card">
            <WhatIf listing={submitted} features={options.whatif_features} />
          </section>
        </>
      )}

      <section className="card map-card">
        <PriceMap
          selectedCentroid={options.neigh_centroids[form.neighbourhood]}
          selectedName={form.neighbourhood}
        />
      </section>
    </div>
  )
}
