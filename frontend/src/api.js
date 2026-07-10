// Thin wrappers around the FastAPI backend. Vite proxies /api → localhost:8000.

export async function fetchOptions() {
  const res = await fetch('/api/options')
  if (!res.ok) throw new Error('Failed to load options')
  return res.json()
}

export async function fetchMapPoints() {
  const res = await fetch('/api/map')
  if (!res.ok) throw new Error('Failed to load map data')
  return res.json()
}

export async function predictPrice(listing) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(listing),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Prediction failed: ${detail}`)
  }
  return res.json()
}

export async function fetchDistribution() {
  const res = await fetch('/api/distribution')
  if (!res.ok) throw new Error('Failed to load distribution')
  return res.json()
}

export async function fetchWhatIf(listing, vary) {
  const res = await fetch(`/api/whatif?vary=${encodeURIComponent(vary)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(listing),
  })
  if (!res.ok) throw new Error('What-if failed')
  return res.json()
}
