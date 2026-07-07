import React, { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { api } from '../services/api'

const POLL_MS = 1000
const ROUTE_COLORS = ['#ff9900', '#f85149', '#d29922', '#56d364']
const DIRECTION_COLORS = { straight: '#4da6ff', left: '#ff9900', right: '#56d364', u_turn: '#f85149' }

const iconNormal = L.divIcon({
  html: '<div style="width:10px;height:10px;border-radius:50%;background:#4da6ff;border:1px solid #fff;"></div>',
  iconSize: [10, 10], className: '',
})
const iconMarked = L.divIcon({
  html: '<div style="width:14px;height:14px;border-radius:50%;background:#f85149;border:2px solid #ff9900;box-shadow:0 0 8px #f85149;"></div>',
  iconSize: [14, 14], className: '',
})
const iconCamera = L.divIcon({
  html: '<div style="width:12px;height:12px;background:#8b5cf6;border:1px solid #fff;transform:rotate(45deg);"></div>',
  iconSize: [12, 12], className: '',
})

export default function MapView() {
  const navigate = useNavigate()
  const mapDivRef = useRef(null)
  const mapRef = useRef(null)
  const layersRef = useRef({ tracks: null, route: null, cameras: null })
  const centeredRef = useRef(false)

  const [state, setState] = useState(null)
  const [error, setError] = useState(null)

  // Init Leaflet map once
  useEffect(() => {
    // View khoi tao ngay tai nga tu Pho Hue - Tran Khat Chan (giam "nhay" ban do
    // truoc khi map state ve). Se duoc fitBounds sat hon o effect ben duoi.
    const map = L.map(mapDivRef.current, { preferCanvas: true }).setView([21.0085, 105.8514], 17)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap', maxZoom: 19,
    }).addTo(map)

    layersRef.current = {
      tracks: L.layerGroup().addTo(map),
      route: L.layerGroup().addTo(map),
      cameras: L.layerGroup().addTo(map),
    }
    mapRef.current = map

    return () => map.remove()
  }, [])

  // Poll map state
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const data = await api.getMapState()
        if (!cancelled) { setState(data); setError(null) }
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    }
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Redraw markers/route whenever state changes
  useEffect(() => {
    if (!state || !state.enabled || !mapRef.current) return
    const map = mapRef.current
    const { tracks: trackLayer, route: routeLayer, cameras: camLayer } = layersRef.current
    trackLayer.clearLayers()
    routeLayer.clearLayers()
    camLayer.clearLayers()

    // Focus dung nga tu MOT LAN: fitBounds om khung camera + tat ca xe co lat/lon
    // (zoom toi da 18 de khong zoom qua sat khi chi co 1 diem). Khung nay chinh la
    // vung giao lo dang co phuong tien -> ban do focus dung nga tu.
    if (!centeredRef.current) {
      const pts = []
      ;(state.cameras || []).forEach(c => { if (c.lat != null) pts.push([c.lat, c.lon]) })
      ;(state.tracks || []).forEach(t => { if (t.lat != null) pts.push([t.lat, t.lon]) })
      if (pts.length === 1) {
        map.setView(pts[0], 18); centeredRef.current = true
      } else if (pts.length > 1) {
        map.fitBounds(L.latLngBounds(pts), { maxZoom: 18, padding: [40, 40] })
        centeredRef.current = true
      }
    }

    ;(state.cameras || []).forEach(cam => {
      L.marker([cam.lat, cam.lon], { icon: iconCamera })
        .bindTooltip(cam.camera_id, { permanent: false })
        .addTo(camLayer)
    })

    ;(state.tracks || []).forEach(t => {
      if (t.lat == null) return
      const icon = t.is_marked ? iconMarked : iconNormal
      const marker = L.marker([t.lat, t.lon], { icon }).addTo(trackLayer)
      const dirTxt = t.direction ? ` — ${t.direction} (${Math.round((t.confidence ?? 0) * 100)}%)` : ''
      marker.bindTooltip(`#${t.global_id} ${t.class}${dirTxt}`, { permanent: false })
      if (!t.is_marked) {
        marker.bindPopup(
          `<b>#${t.global_id} (${t.class})</b><br>` +
          `<button id="mark-btn-${t.global_id}" ` +
          `style="margin-top:6px;padding:4px 10px;background:#f85149;color:#fff;` +
          `border:none;border-radius:4px;cursor:pointer;font-size:0.8rem;">Đánh dấu theo dõi</button>`
        )
        marker.on('popupopen', () => {
          const btn = document.getElementById(`mark-btn-${t.global_id}`)
          if (btn) btn.onclick = () => { api.markObject(t.global_id).catch(() => {}); map.closePopup() }
        })
      }
    })

    const marked = state.marked
    if (marked && marked.lat != null) {
      const hops = marked.route || []
      const pts = [[marked.lat, marked.lon], ...hops.filter(h => h.lat != null).map(h => [h.lat, h.lon])]
      if (pts.length > 1) {
        L.polyline(pts, { color: ROUTE_COLORS[0], weight: 4, opacity: 0.85, dashArray: '8 4' }).addTo(routeLayer)
      }
      hops.forEach((wp, i) => {
        if (wp.lat == null) return
        L.circleMarker([wp.lat, wp.lon], {
          radius: 6, color: ROUTE_COLORS[i % ROUTE_COLORS.length],
          fillColor: ROUTE_COLORS[i % ROUTE_COLORS.length], fillOpacity: 0.9, weight: 2,
        }).bindTooltip(`Hop ${wp.hop} (${wp.direction_used})`, { permanent: false }).addTo(routeLayer)
      })
    }
  }, [state])

  const unmark = () => { api.unmarkObject().catch(() => {}) }

  if (state && !state.enabled) {
    return (
      <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center text-center p-6">
        <button onClick={() => navigate('/')} className="absolute top-4 left-4 text-gray-400 hover:text-white text-sm">
          ← Dashboard
        </button>
        <div className="text-4xl mb-3">🗺️</div>
        <h1 className="text-white text-lg font-bold mb-2">Map / Route Prediction chưa khả dụng</h1>
        <p className="text-gray-400 text-sm max-w-md">
          {state.message || 'Nguồn hiện tại chưa được cấu hình route_prediction (map georeference).'}
        </p>
      </div>
    )
  }

  const marked = state?.marked
  const probs = marked?.probabilities || {}

  return (
    <div className="flex flex-col h-screen bg-gray-950">
      <header className="flex items-center justify-between px-4 py-2.5 bg-gray-900 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white text-sm">
            ← Dashboard
          </button>
          <span className="text-lg font-bold text-white tracking-wide">Bản đồ & Dự đoán lộ trình</span>
        </div>
        {error && <span className="text-xs text-red-400">Lỗi tải dữ liệu: {error}</span>}
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div ref={mapDivRef} className="flex-1" />

        <div className="w-[300px] shrink-0 bg-gray-900 border-l border-gray-800 overflow-y-auto p-3 text-sm">
          <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Đối tượng đang theo dõi</div>
          {!marked ? (
            <div className="text-gray-500 text-xs">
              Click vào 1 xe trên bản đồ và chọn "Đánh dấu theo dõi" để dự đoán lộ trình dài hạn.
            </div>
          ) : (
            <>
              <div className="flex justify-between border-b border-gray-800 py-1">
                <span className="text-gray-500">Track ID</span>
                <span className="text-white font-medium">#{marked.global_id}</span>
              </div>
              <div className="flex justify-between border-b border-gray-800 py-1">
                <span className="text-gray-500">Hướng dự đoán</span>
                <span className="text-white font-medium">{marked.direction}</span>
              </div>
              <div className="flex justify-between border-b border-gray-800 py-1">
                <span className="text-gray-500">Độ tin cậy</span>
                <span className="text-white font-medium">{Math.round((marked.confidence ?? 0) * 100)}%</span>
              </div>
              <button
                onClick={unmark}
                className="mt-3 w-full text-xs px-2 py-1.5 rounded bg-gray-700 hover:bg-red-700 transition-colors text-white"
              >
                Bỏ đánh dấu
              </button>

              <div className="text-gray-500 text-xs uppercase tracking-wider mt-4 mb-2">Xác suất hướng đi</div>
              {['straight', 'left', 'right', 'u_turn'].map(d => (
                <div key={d} className="mb-2">
                  <div className="flex justify-between mb-0.5">
                    <span style={{ color: DIRECTION_COLORS[d] }}>{d}</span>
                    <span className="text-gray-300">{Math.round((probs[d] ?? 0) * 100)}%</span>
                  </div>
                  <div className="bg-gray-800 rounded h-1.5">
                    <div
                      className="h-1.5 rounded"
                      style={{ width: `${Math.round((probs[d] ?? 0) * 100)}%`, background: DIRECTION_COLORS[d] }}
                    />
                  </div>
                </div>
              ))}

              <div className="text-gray-500 text-xs uppercase tracking-wider mt-4 mb-2">Lộ trình dự đoán</div>
              {(marked.route || []).length === 0 ? (
                <div className="text-gray-500 text-xs">Chưa có dữ liệu lộ trình.</div>
              ) : (
                marked.route.map(wp => (
                  <div key={wp.hop} className="border-b border-gray-800 py-1">
                    <div className="text-green-400 font-medium">Hop {wp.hop} — road {wp.road_id}</div>
                    <div className="text-gray-500 text-xs">
                      {wp.lat?.toFixed(5)}, {wp.lon?.toFixed(5)} | hdg {wp.heading_deg?.toFixed(0)}°
                    </div>
                  </div>
                ))
              )}
            </>
          )}

          <div className="text-gray-500 text-xs uppercase tracking-wider mt-4 mb-2">
            Phương tiện ({(state?.tracks || []).length})
          </div>
          {(state?.tracks || []).map(t => (
            <div
              key={t.global_id}
              onClick={() => api.markObject(t.global_id).catch(() => {})}
              className={`flex justify-between py-1 border-b border-gray-800 cursor-pointer hover:bg-gray-800/50
                ${t.is_marked ? 'text-red-400' : 'text-gray-300'}`}
            >
              <span>#{t.global_id} {t.class}</span>
              <span className="text-gray-500">{t.direction ?? '—'}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
