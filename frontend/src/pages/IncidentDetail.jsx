import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'

const SEVERITY_COLOR = {
  CRITICAL: 'text-red-400',
  WARNING:  'text-yellow-400',
  INFO:     'text-blue-400',
}

export default function IncidentDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [track, setTrack] = useState(null)

  // id format: TYPE_gid_timestamp — trích global_id
  const globalId = id?.split('_').find(p => /^\d+$/.test(p))

  useEffect(() => {
    if (!globalId) return
    api.getTrackById(globalId)
      .then(setTrack)
      .catch(() => {})
  }, [globalId])

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <button
        onClick={() => navigate(-1)}
        className="text-gray-400 hover:text-white text-sm mb-6 flex items-center gap-1"
      >
        ← Quay lại
      </button>

      <h1 className="text-xl font-bold text-white mb-6">Chi tiết sự cố</h1>

      {/* Incident info từ URL params */}
      <div className="bg-gray-900 rounded-xl p-5 border border-gray-800 mb-4">
        <div className="text-sm text-gray-400 mb-1">ID sự cố</div>
        <div className="font-mono text-white text-sm">{id}</div>
      </div>

      {/* Track info */}
      {track && (
        <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
          <h2 className="text-base font-semibold text-white mb-4">
            Đối tượng #{globalId}
          </h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-gray-400 mb-1">Loại</div>
              <div className="text-white">{track.class ?? '--'}</div>
            </div>
            <div>
              <div className="text-gray-400 mb-1">Trạng thái</div>
              <div className="text-white">{track.status ?? '--'}</div>
            </div>
            <div>
              <div className="text-gray-400 mb-1">Lần đầu xuất hiện</div>
              <div className="text-white">
                {track.first_seen ? new Date(track.first_seen).toLocaleString('vi-VN') : '--'}
              </div>
            </div>
            <div>
              <div className="text-gray-400 mb-1">Lần cuối thấy</div>
              <div className="text-white">
                {track.last_seen ? new Date(track.last_seen).toLocaleString('vi-VN') : '--'}
              </div>
            </div>
          </div>

          {/* Timeline cameras */}
          {track.camera_history && (
            <div className="mt-5">
              <div className="text-gray-400 text-sm mb-2">Di chuyển xuyên camera</div>
              <div className="flex items-center gap-2 flex-wrap">
                {Object.keys(track.camera_history).map((cam, i, arr) => (
                  <React.Fragment key={cam}>
                    <span className="bg-blue-800 text-blue-200 text-xs px-2 py-1 rounded font-mono">
                      {cam}
                    </span>
                    {i < arr.length - 1 && (
                      <span className="text-gray-600 text-sm">→</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
