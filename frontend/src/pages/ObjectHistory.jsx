import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../services/api'

export default function ObjectHistory() {
  const { global_id } = useParams()
  const navigate = useNavigate()
  const [track, setTrack] = useState(null)

  useEffect(() => {
    api.getTrackById(global_id)
      .then(setTrack)
      .catch(() => {})
  }, [global_id])

  if (!track) return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center text-gray-600">
      Đang tải...
    </div>
  )

  const cameras = track.camera_history ? Object.keys(track.camera_history) : []

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <button onClick={() => navigate(-1)} className="text-gray-400 hover:text-white text-sm mb-6">
        ← Quay lại
      </button>

      <h1 className="text-xl font-bold text-white mb-6">
        Đối tượng #{global_id}
        <span className="ml-3 text-sm font-normal text-gray-400">{track.class}</span>
      </h1>

      {/* Thông tin cơ bản */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Lần đầu xuất hiện', value: track.first_seen ? new Date(track.first_seen).toLocaleString('vi-VN') : '--' },
          { label: 'Lần cuối thấy',    value: track.last_seen  ? new Date(track.last_seen).toLocaleString('vi-VN')  : '--' },
          { label: 'Số camera đi qua', value: cameras.length },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-xs text-gray-400 mb-1">{label}</div>
            <div className="text-white font-semibold">{value}</div>
          </div>
        ))}
      </div>

      {/* Timeline xuyên camera */}
      <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Hành trình xuyên camera</h2>
        <div className="flex items-center gap-3 flex-wrap">
          {cameras.map((cam, i) => (
            <React.Fragment key={cam}>
              <div className="text-center">
                <div className="bg-blue-900 border border-blue-600 rounded-lg px-4 py-2">
                  <div className="font-mono text-blue-200 font-bold">{cam}</div>
                  <div className="text-xs text-blue-400 mt-0.5">
                    {track.camera_history[cam]?.length ?? 0} điểm
                  </div>
                </div>
              </div>
              {i < cameras.length - 1 && (
                <div className="text-gray-500 text-lg">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  )
}
