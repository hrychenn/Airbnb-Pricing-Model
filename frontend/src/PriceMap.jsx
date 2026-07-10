import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from 'react-leaflet'
import { fetchMapPoints } from './api'

// Continuous blue → teal → yellow → red colour scale for price
const STOPS = [
  [0.0, [37, 52, 148]],
  [0.35, [44, 127, 184]],
  [0.55, [127, 205, 187]],
  [0.72, [255, 221, 120]],
  [0.86, [253, 141, 60]],
  [1.0, [227, 26, 28]],
]

function colorFor(price, lo, hi) {
  const t = Math.max(0, Math.min(1, (price - lo) / (hi - lo)))
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1]
      const [t1, c1] = STOPS[i]
      const f = (t - t0) / (t1 - t0)
      const rgb = c0.map((c, k) => Math.round(c + f * (c1[k] - c)))
      return `rgb(${rgb.join(',')})`
    }
  }
  return `rgb(${STOPS[STOPS.length - 1][1].join(',')})`
}

// Recenters the map when the selected neighbourhood changes
function Recenter({ centroid }) {
  const map = useMap()
  useEffect(() => {
    if (centroid) map.setView(centroid, 13, { animate: true })
  }, [centroid, map])
  return null
}

// The map container can measure zero on first paint (in dev, CSS is injected
// asynchronously), which leaves the vector overlay stale. Re-measure once mounted.
function InvalidateOnMount() {
  const map = useMap()
  useEffect(() => {
    const id = setTimeout(() => map.invalidateSize(), 250)
    return () => clearTimeout(id)
  }, [map])
  return null
}

function Legend({ lo, hi }) {
  const gradient = STOPS.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`).join(', ')
  return (
    <div className="map-legend">
      <span>${Math.round(lo)}</span>
      <div className="legend-bar" style={{ background: `linear-gradient(90deg, ${gradient})` }} />
      <span>${Math.round(hi)}+</span>
    </div>
  )
}

export default function PriceMap({ selectedCentroid, selectedName }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchMapPoints().then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="error-banner">Map failed to load: {error}</div>
  if (!data) return <div className="loading">Loading map…</div>

  const { points, price_lo, price_hi } = data

  return (
    <div className="map-wrap">
      <div className="map-head">
        <h3>NYC listing prices</h3>
        <Legend lo={price_lo} hi={price_hi} />
      </div>
      <MapContainer center={[40.7128, -73.98]} zoom={11} className="map" scrollWheelZoom={false}>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap &copy; CARTO'
        />
        {points.map((p, i) => (
          <CircleMarker
            key={i}
            center={[p.lat, p.lng]}
            radius={3}
            pathOptions={{ color: colorFor(p.price, price_lo, price_hi), weight: 0, fillOpacity: 0.55 }}
          />
        ))}
        {selectedCentroid && (
          <CircleMarker
            center={selectedCentroid}
            radius={11}
            pathOptions={{ color: '#111', weight: 3, fillOpacity: 0 }}
          >
            <Tooltip permanent direction="top" offset={[0, -8]}>{selectedName}</Tooltip>
          </CircleMarker>
        )}
        <Recenter centroid={selectedCentroid} />
        <InvalidateOnMount />
      </MapContainer>
      <p className="foot">Each dot is a listing, coloured by nightly price (blue = cheaper, red = pricier). The ring marks your selected neighbourhood.</p>
    </div>
  )
}
