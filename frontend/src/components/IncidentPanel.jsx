import React, { useState } from 'react'
import IncidentCard from './IncidentCard'

const SEVERITY_ORDER = { CRITICAL: 0, WARNING: 1, INFO: 2 }

export default function IncidentPanel({ incidents, unreadCount, onAcknowledge }) {
  const [filter, setFilter] = useState('ALL')

  const filtered = incidents
    .filter(i => filter === 'ALL' || i.severity === filter)
    .sort((a, b) =>
      (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
    )

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-xl border border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-bold text-white">Sự cố</h2>
          {unreadCount > 0 && (
            <span className="bg-red-600 text-white text-xs font-bold px-1.5 py-0.5 rounded-full">
              {unreadCount}
            </span>
          )}
        </div>
        {/* Severity filter */}
        <div className="flex gap-1">
          {['ALL', 'CRITICAL', 'WARNING', 'INFO'].map(s => (
            <button
              key={s}
              onClick={() => setFilter(s)}
              className={`text-xs px-2 py-1 rounded transition-colors
                ${filter === s
                  ? 'bg-gray-600 text-white'
                  : 'text-gray-400 hover:text-white'}`}
            >
              {s === 'ALL' ? 'Tất cả' : s}
            </button>
          ))}
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filtered.length === 0 ? (
          <div className="text-center text-gray-600 text-sm mt-8">
            Không có sự cố nào
          </div>
        ) : (
          filtered.map(inc => (
            <IncidentCard
              key={inc.id}
              incident={inc}
              onAcknowledge={onAcknowledge}
            />
          ))
        )}
      </div>
    </div>
  )
}
