import React from 'react'
import { useNavigate } from 'react-router-dom'

const SEVERITY_STYLE = {
  CRITICAL: { bar: 'bg-red-600',    badge: 'bg-red-600',    icon: '🚨' },
  WARNING:  { bar: 'bg-yellow-500', badge: 'bg-yellow-500 text-black', icon: '⚠' },
  INFO:     { bar: 'bg-blue-600',   badge: 'bg-blue-600',   icon: 'ℹ' },
}

export default function IncidentCard({ incident, onAcknowledge }) {
  const navigate = useNavigate()
  const style    = SEVERITY_STYLE[incident.severity] ?? SEVERITY_STYLE.INFO
  const ts       = new Date(incident.timestamp).toLocaleTimeString('vi-VN')

  return (
    <div className={`relative bg-gray-800 rounded-lg overflow-hidden border border-gray-700
                     ${incident.acknowledged ? 'opacity-50' : ''}`}>
      {/* Severity bar bên trái */}
      <div className={`absolute left-0 top-0 bottom-0 w-1 ${style.bar}`} />

      <div className="pl-4 pr-3 py-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs px-1.5 py-0.5 rounded font-bold ${style.badge}`}>
                {style.icon} {incident.severity}
              </span>
              <span className="text-xs text-gray-400 font-mono">{ts}</span>
              <span className="text-xs text-gray-500">{incident.camera_id}</span>
            </div>

            {/* Type + message */}
            <p className="text-sm font-semibold text-white truncate">
              {incident.type.replace(/_/g, ' ')}
            </p>
            <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">
              {incident.message}
            </p>
          </div>

          {/* Object ID */}
          <div className="text-right shrink-0">
            <div className="text-xs text-gray-500">Object</div>
            <div className="text-sm font-mono font-bold text-white">
              #{incident.global_id}
            </div>
          </div>
        </div>

        {/* Actions */}
        {!incident.acknowledged && (
          <div className="flex gap-2 mt-2">
            <button
              onClick={() => navigate(`/incident/${incident.id}`)}
              className="flex-1 text-xs py-1 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
            >
              Chi tiết
            </button>
            <button
              onClick={() => onAcknowledge(incident.id)}
              className="flex-1 text-xs py-1 rounded bg-gray-700 hover:bg-green-700 transition-colors"
            >
              Xác nhận
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
