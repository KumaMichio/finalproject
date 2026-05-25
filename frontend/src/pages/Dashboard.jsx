import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CameraGrid from '../components/CameraGrid'
import IncidentPanel from '../components/IncidentPanel'
import StatsBar from '../components/StatsBar'
import { useIncidents } from '../hooks/useIncidents'
import { api } from '../services/api'

export default function Dashboard() {
  const [cameras, setCameras] = useState([])
  const { incidents, acknowledge, unreadCount, connected } = useIncidents()
  const navigate = useNavigate()

  useEffect(() => {
    api.getCameras()
      .then(data => setCameras(Array.isArray(data) ? data : []))
      .catch(() => {})
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
        <div className="w-[360px] shrink-0">
          <IncidentPanel
            incidents={incidents}
            unreadCount={unreadCount}
            onAcknowledge={acknowledge}
          />
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar connected={connected} />
    </div>
  )
}
