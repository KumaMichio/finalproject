const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json()
}

async function put(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PUT ${path} → ${res.status}`)
  return res.json()
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `POST ${path} → ${res.status}`)
  return data
}

export const api = {
  getCameras:    ()       => get('/cameras'),
  getStats:      ()       => get('/stats'),
  getAlerts:     (params) => get(`/alerts${params ? '?' + new URLSearchParams(params) : ''}`),
  getTracks:     ()       => get('/tracks'),
  getTrackById:  (id)     => get(`/tracks/${id}`),
  acknowledgeAlert: (id)  => put(`/alerts/${id}/acknowledge`, {}),
  streamUrl:     (camId)  => `/stream/${camId}`,
  getScenarios:  ()       => get('/scenarios'),
  triggerScenario: (name) => post(`/scenarios/${name}`),
}
