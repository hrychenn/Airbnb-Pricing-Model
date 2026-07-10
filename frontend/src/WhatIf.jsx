import { useEffect, useState } from 'react'
import { fetchWhatIf } from './api'

const W = 560
const H = 220
const PAD = { l: 52, r: 16, t: 16, b: 34 }

export default function WhatIf({ listing, features }) {
  const [vary, setVary] = useState('accommodates')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let alive = true
    setLoading(true); setError(null)
    fetchWhatIf(listing, vary)
      .then((d) => { if (alive) setData(d) })
      .catch((e) => { if (alive) setError(e.message) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
    // re-fetch when the swept feature changes or the listing is re-predicted
  }, [vary, listing])

  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b

  let body = null
  if (error) body = <div className="error-banner">{error}</div>
  else if (!data) body = <div className="loading">Loading…</div>
  else {
    const pts = data.points
    const xs = pts.map((p) => p.x)
    const ys = pts.map((p) => p.price)
    const xMin = Math.min(...xs), xMax = Math.max(...xs)
    const yMin = Math.min(...ys), yMax = Math.max(...ys)
    const yLo = Math.floor((yMin * 0.9) / 10) * 10
    const yHi = Math.ceil((yMax * 1.05) / 10) * 10
    const xFor = (x) => PAD.l + ((x - xMin) / (xMax - xMin || 1)) * plotW
    const yFor = (y) => PAD.t + plotH - ((y - yLo) / (yHi - yLo || 1)) * plotH
    const line = pts.map((p, i) => `${i ? 'L' : 'M'}${xFor(p.x).toFixed(1)},${yFor(p.price).toFixed(1)}`).join(' ')
    const area = `${line} L${xFor(xMax).toFixed(1)},${(PAD.t + plotH).toFixed(1)} L${xFor(xMin).toFixed(1)},${(PAD.t + plotH).toFixed(1)} Z`

    // marker at the listing's current value (nearest swept point)
    const cur = data.current
    const nearest = pts.reduce((a, b) => (Math.abs(b.x - cur) < Math.abs(a.x - cur) ? b : a), pts[0])

    const yTicks = [yLo, Math.round((yLo + yHi) / 2), yHi]
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} className="wi-svg" preserveAspectRatio="xMidYMid meet">
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={PAD.l} y1={yFor(v)} x2={W - PAD.r} y2={yFor(v)} className="wi-grid" />
            <text x={PAD.l - 8} y={yFor(v) + 4} className="wi-tick" textAnchor="end">${v}</text>
          </g>
        ))}
        <path d={area} className="wi-area" />
        <path d={line} className="wi-line" />
        {pts.map((p, i) => <circle key={i} cx={xFor(p.x)} cy={yFor(p.price)} r="3" className="wi-dot" />)}
        {/* highlight current value */}
        <line x1={xFor(nearest.x)} y1={PAD.t} x2={xFor(nearest.x)} y2={PAD.t + plotH} className="wi-cur" />
        <circle cx={xFor(nearest.x)} cy={yFor(nearest.price)} r="5.5" className="wi-cur-dot" />
        {pts.map((p, i) => (
          <text key={i} x={xFor(p.x)} y={H - 10} className="wi-tick" textAnchor="middle">{p.x}</text>
        ))}
      </svg>
    )
  }

  return (
    <div className="whatif">
      <div className="whatif-head">
        <h3>What-if: how price responds to…</h3>
        <select value={vary} onChange={(e) => setVary(e.target.value)}>
          {features.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
        </select>
      </div>
      <div className={loading ? 'wi-body dim' : 'wi-body'}>{body}</div>
      <p className="foot">Predicted price as <strong>{data?.label ?? '…'}</strong> varies, holding every other field at your current values. The highlighted point is your listing.</p>
    </div>
  )
}
