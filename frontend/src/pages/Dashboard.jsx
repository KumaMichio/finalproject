import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CameraGrid from '../components/CameraGrid'
import IncidentPanel from '../components/IncidentPanel'
import ScenarioPanel from '../components/ScenarioPanel'
import StatsBar from '../components/StatsBar'
import { useIncidents } from '../hooks/useIncidents'
import { api } from '../services/api'

export default function Dashboard() {
  const [cameras, setCameras] = useState([])
  const [source, setSource] = useState('carla')
  const [mapEnabled, setMapEnabled] = useState(false)
  const { incidents, acknowledge, unreadCount, connected } = useIncidents()
  const navigate = useNavigate()

  useEffect(() => {
    api.getCameras()
      .then(data => setCameras(Array.isArray(data) ? data : []))
      .catch(() => {})

    // source='carla' -> co the kich hoat kich ban dung san (ScenarioPanel);
    // source='file' (video CCTV thuc) -> khong co CARLA de dung kich ban,
    // an panel di thay vi hien nut bam luon bao loi.
    // map_enabled -> RoutePredictor/GeoReference da duoc cau hinh (config co
    // section 'route_prediction'), hien nut "Bản đồ" de xem du doan lo trinh.
    api.getStats()
      .then(data => {
        setSource(data?.source ?? 'carla')
        setMapEnabled(Boolean(data?.map_enabled))
      })
      .catch(() => {})

    // Xoa lich su su co cu moi khi tai lai trang, de bat dau phien demo moi
    api.clearAlerts().catch(() => {})
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gray-950">
      {/* Top bar */}
      <header className="flex items-center justify-between px-4 py-2.5 bg-gray-900 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold text-white tracking-wide">CCTV AI</span>
          <span className="text-xs text-gray-500">Hệ thống giám sát thông minh</span>
        </div>
        <div className="flex items-center gap-3">
          {mapEnabled && (
            <button
              onClick={() => navigate('/map')}
              className="flex items-center gap-1.5 bg-purple-600 hover:bg-purple-700 text-white text-xs px-3 py-1.5 rounded-full transition-colors"
            >
              🗺️ Bản đồ & lộ trình
            </button>
          )}
          {unreadCount > 0 && (
            <button
              onClick={() => navigate('/alerts')}
              className="flex items-center gap-1.5 bg-red-600 hover:bg-red-700 text-white text-xs px-3 py-1.5 rounded-full transition-colors"
            >
              🚨 {unreadCount} sự cố mới
            </button>
          )}
          <button
            onClick={() => navigate('/alerts')}
            className="text-xs text-gray-400 hover:text-white transition-colors"
          >
            Lịch sử
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 gap-3 p-3 overflow-hidden">
        {/* Camera grid — chiếm 65% */}
        <div className="flex-1 min-w-0">
          {cameras.length > 0 ? (
            <CameraGrid
              cameras={cameras}
              incidents={incidents}
              onCameraClick={(cam) => navigate(`/alerts?camera=${cam.id}`)}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-600">
              <div className="text-center">
                <div className="text-4xl mb-2">📷</div>
                <div className="text-sm">Đang kết nối camera...</div>
              </div>
            </div>
          )}
        </div>

        {/* Incident panel — cố định 360px */}
        <div className="w-[360px] shrink-0 flex flex-col gap-3">
          <div className="flex-1 min-h-0">
            <IncidentPanel
              incidents={incidents}
              unreadCount={unreadCount}
              onAcknowledge={acknowledge}
            />
          </div>
          {source === 'carla' && <ScenarioPanel />}
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar connected={connected} />
    </div>
  )
}
