import { useEffect, useState } from 'react'
import { fetchDistribution } from './api'

// Exact-ish percentile via linear interpolation over the CDF returned by the API
function percentileOf(price, cdf) {
  if (price <= cdf[0][0]) return 0
  if (price >= cdf[cdf.length - 1][0]) return 100
  for (let i = 1; i < cdf.length; i++) {
    const [p0, q0] = cdf[i - 1]
    const [p1, q1] = cdf[i]
    if (price <= p1) {
      const f = (price - p0) / (p1 - p0 || 1)
      return Math.round((q0 + f * (q1 - q0)) * 100)
    }
  }
  return 100
}

const W = 560
const H = 200
const PAD = { l: 8, r: 8, t: 10, b: 26 }

export default function Distribution({ price }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDistribution().then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="error-banner">{error}</div>
  if (!data) return <div className="loading">Loading distribution…</div>

  const { bins, cdf, max } = data
  const maxCount = Math.max(...bins.map((b) => b.count))
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const lo = bins[0].x0
  const hi = bins[bins.length - 1].x1
  const xFor = (v) => PAD.l + ((Math.min(v, hi) - lo) / (hi - lo)) * plotW
  const barW = plotW / bins.length

  const pct = percentileOf(price, cdf)
  const topPct = 100 - pct
  const markX = xFor(price)
  const ticks = [lo, 250, 500, 750, `${max}+`]
  const tickVals = [lo, 250, 500, 750, max]

  return (
    <div className="dist">
      <div className="dist-caption">
        Your <strong>${Math.round(price).toLocaleString()}</strong> sits at the{' '}
        <strong>{pct}th percentile</strong> — pricier than {pct}% of NYC listings
        {topPct <= 50 ? ` (top ${topPct}%)` : ''}.
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="dist-svg" preserveAspectRatio="xMidYMid meet">
        {bins.map((b, i) => {
          const h = (b.count / maxCount) * plotH
          const below = b.x1 <= price
          return (
            <rect
              key={i}
              x={PAD.l + i * barW + 0.5}
              y={PAD.t + plotH - h}
              width={barW - 1}
              height={h}
              rx="1.5"
              className={below ? 'dist-bar below' : 'dist-bar'}
            />
          )
        })}
        {/* your-listing marker */}
        <line x1={markX} y1={PAD.t - 4} x2={markX} y2={PAD.t + plotH} className="dist-mark" />
        <circle cx={markX} cy={PAD.t - 4} r="4" className="dist-mark-dot" />
        {/* x-axis ticks */}
        {tickVals.map((v, i) => (
          <text key={i} x={xFor(v)} y={H - 8} className="dist-tick" textAnchor="middle">
            ${ticks[i]}
          </text>
        ))}
      </svg>
    </div>
  )
}
