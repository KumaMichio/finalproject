import React, { useState } from 'react'
import { api } from '../services/api'

const SCENARIOS = [
  { name: 'hit_and_run', label: 'Tai nạn rồi bỏ chạy', icon: '🚗💨' },
  { name: 'pedestrian_hit', label: 'Đâm người đi bộ', icon: '🚶‍♂️💥' },
  { name: 'red_light_crash', label: 'Vượt đèn đỏ đâm nhau', icon: '🚦💥' },
  { name: 'rear_end', label: 'Đâm từ phía sau', icon: '🚙💢' },
  { name: 'sudden_stop', label: 'Dừng đột ngột', icon: '🛑' },
]

export default function ScenarioPanel() {
  const [pending, setPending] = useState(null)
  const [feedback, setFeedback] = useState(null)

  const trigger = async (name) => {
    setPending(name)
    setFeedback(null)
    try {
      const res = await api.triggerScenario(name)
      setFeedback({ ok: true, message: res.message, cameraId: res.camera_id })
    } catch (e) {
      setFeedback({ ok: false, message: e.message })
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="flex flex-col bg-gray-900 rounded-xl border border-gray-800 p-3 gap-2">
      <h2 className="text-sm font-bold text-white">Kịch bản demo</h2>
      <div className="grid grid-cols-1 gap-1.5">
        {SCENARIOS.map(s => (
          <button
            key={s.name}
            onClick={() => trigger(s.name)}
            disabled={pending !== null}
            className="flex items-center gap-2 text-xs px-2.5 py-1.5 rounded
              bg-gray-800 hover:bg-gray-700 text-gray-200 disabled:opacity-50
              transition-colors text-left"
          >
            <span>{s.icon}</span>
            <span className="flex-1">{s.label}</span>
            {pending === s.name && <span className="text-gray-500">...</span>}
          </button>
        ))}
      </div>
      {feedback && (
        <div className={`text-xs px-2 py-1.5 rounded ${
          feedback.ok ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
        }`}>
          {feedback.message}
        </div>
      )}
    </div>
  )
}
