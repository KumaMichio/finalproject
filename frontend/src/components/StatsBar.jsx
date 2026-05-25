import React, { useEffect, useState } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'

export default function StatsBar({ connected }) {
  const [stats, setStats] = useState({ fps: 0, active_tracks: 0, alert_count: 0 })
  const { lastMessage } = useWebSocket('/ws/stats')

  useEffect(() => {
    if (!lastMessage) return
    try {
      const data = JSON.parse(lastMessage.data)
      setStats(prev => ({ ...prev, ...data }))
    } catch {}
  }, [lastMessage])

  const dot = connected
    ? 'bg-green-400 animate-pulse'
    : 'bg-red-500'

  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-gray-900 border-t border-gray-800 text-xs text-gray-400">
      <div className="flex items-center gap-1.5">
        <div className={`w-2 h-2 rounded-full ${dot}`} />
        <span className={connected ? 'text-green-400' : 'text-red-400'}>
          {connected ? 'ONLINE' : 'OFFLINE'}
        </span>
      </div>
      <span>FPS: <b className="text-white">{stats.fps?.toFixed(1) ?? '--'}</b></span>
      <span>Đối tượng: <b className="text-white">{stats.active_tracks ?? '--'}</b></span>
      <span>Sự cố hôm nay: <b className="text-white">{stats.alert_count ?? '--'}</b></span>
      <span className="ml-auto font-mono">
        {new Date().toLocaleTimeString('vi-VN')}
      </span>
    </div>
  )
}
