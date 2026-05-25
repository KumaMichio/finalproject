import React from 'react'
import { api } from '../services/api'

const SEVERITY_BORDER = {
  CRITICAL: 'border-red-500 shadow-red-500/40',
  WARNING:  'border-yellow-400 shadow-yellow-400/30',
  INFO:     'border-blue-500',
}

/**
 * Hiển thị 1 camera: MJPEG live feed + badge sự cố gần nhất.
 */
export default function CameraFeed({ camera, latestIncident, onClick }) {
  const borderClass = latestIncident
    ? SEVERITY_BORDER[latestIncident.severity] ?? 'border-gray-700'
    : 'border-gray-700'

  return (
    <div
      className={`relative bg-gray-900 rounded-lg overflow-hidden border-2 shadow-lg cursor-pointer
                  transition-all duration-300 hover:scale-[1.02] ${borderClass}`}
      onClick={() => onClick?.(camera)}
    >
      {/* MJPEG stream */}
      <img
        src={api.streamUrl(camera.id)}
        alt={camera.name}
        className="w-full aspect-video object-cover"
        onError={(e) => { e.target.style.display = 'none' }}
      />

      {/* Camera label */}
      <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono">
        {camera.id}
      </div>

      {/* Status dot */}
      <div className={`absolute top-2 right-2 w-2.5 h-2.5 rounded-full
        ${camera.status === 'active' ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />

      {/* Incident badge */}
      {latestIncident && (
        <div className={`absolute bottom-0 left-0 right-0 px-2 py-1.5 text-xs font-semibold
          ${latestIncident.severity === 'CRITICAL' ? 'bg-red-600/90' :
            latestIncident.severity === 'WARNING'  ? 'bg-yellow-500/90 text-black' :
                                                     'bg-blue-600/90'}`}>
          {latestIncident.type.replace(/_/g, ' ')} — #{latestIncident.global_id}
        </div>
      )}
    </div>
  )
}
