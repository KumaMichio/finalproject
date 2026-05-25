import { useEffect, useState, useRef } from 'react'
import { useWebSocket } from './useWebSocket'

const SEVERITY_SOUND = {
  CRITICAL: '/sounds/critical.mp3',
  WARNING:  '/sounds/warning.mp3',
}

/**
 * Quản lý danh sách incidents real-time từ WebSocket /ws/alerts.
 * Tự động phát âm thanh và flash màn hình khi có CRITICAL alert.
 */
export function useIncidents() {
  const [incidents, setIncidents] = useState([])
  const { lastMessage, connected } = useWebSocket('/ws/alerts')
  const audioRef = useRef({})

  useEffect(() => {
    if (!lastMessage) return
    let data
    try { data = JSON.parse(lastMessage.data) } catch { return }
    if (data.event !== 'incident') return

    const incident = { ...data, id: `${data.type}_${data.global_id}_${Date.now()}` }

    setIncidents(prev => [incident, ...prev].slice(0, 200))

    // Âm thanh cảnh báo
    const soundPath = SEVERITY_SOUND[data.severity]
    if (soundPath) {
      if (!audioRef.current[data.severity]) {
        audioRef.current[data.severity] = new Audio(soundPath)
      }
      audioRef.current[data.severity].play().catch(() => {})
    }

    // Flash màn hình đỏ cho CRITICAL
    if (data.severity === 'CRITICAL') {
      document.body.classList.add('alert-flash')
      setTimeout(() => document.body.classList.remove('alert-flash'), 500)
    }
  }, [lastMessage])

  const acknowledge = (incidentId) => {
    setIncidents(prev =>
      prev.map(i => i.id === incidentId ? { ...i, acknowledged: true } : i)
    )
  }

  const unreadCount = incidents.filter(i => !i.acknowledged).length

  return { incidents, acknowledge, unreadCount, connected }
}
