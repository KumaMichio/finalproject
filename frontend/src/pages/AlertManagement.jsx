import React, { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../services/api'

const SEVERITY_BADGE = {
  CRITICAL: 'bg-red-600 text-white',
  WARNING:  'bg-yellow-500 text-black',
  INFO:     'bg-blue-600 text-white',
}

export default function AlertManagement() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [alerts, setAlerts]   = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter]   = useState({
    severity: '',
    camera_id: searchParams.get('camera') ?? '',
  })

  useEffect(() => {
    setLoading(true)
    const params = {}
    if (filter.severity)  params.severity  = filter.severity
    if (filter.camera_id) params.camera_id = filter.camera_id

    api.getAlerts(params)
      .then(data => setAlerts(Array.isArray(data) ? data : []))
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false))
  }, [filter])

  const acknowledge = async (alertId) => {
    await api.acknowledgeAlert(alertId)
    setAlerts(prev =>
      prev.map(a => a.id === alertId ? { ...a, status: 'acknowledged' } : a)
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white text-sm">
          ← Dashboard
        </button>
        <h1 className="text-xl font-bold text-white">Quản lý sự cố</h1>
      </div>

      {/* Filter bar */}
      <div className="flex gap-3 mb-4">
        <select
          value={filter.severity}
          onChange={e => setFilter(f => ({ ...f, severity: e.target.value }))}
          className="bg-gray-800 text-white text-sm px-3 py-2 rounded border border-gray-700"
        >
          <option value="">Tất cả mức độ</option>
          <option value="CRITICAL">CRITICAL</option>
          <option value="WARNING">WARNING</option>
          <option value="INFO">INFO</option>
        </select>
        <input
          value={filter.camera_id}
          onChange={e => setFilter(f => ({ ...f, camera_id: e.target.value }))}
          placeholder="Lọc theo camera..."
          className="bg-gray-800 text-white text-sm px-3 py-2 rounded border border-gray-700 w-44"
        />
      </div>

      {/* Table */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-800 text-gray-400 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">Thời gian</th>
              <th className="px-4 py-3 text-left">Loại</th>
              <th className="px-4 py-3 text-left">Mức độ</th>
              <th className="px-4 py-3 text-left">Camera</th>
              <th className="px-4 py-3 text-left">Object</th>
              <th className="px-4 py-3 text-left">Trạng thái</th>
              <th className="px-4 py-3 text-left">Hành động</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {loading ? (
              <tr>
                <td colSpan={7} className="text-center text-gray-600 py-8">Đang tải...</td>
              </tr>
            ) : alerts.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center text-gray-600 py-8">Không có sự cố</td>
              </tr>
            ) : (
              alerts.map(alert => (
                <tr key={alert.id} className="hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-3 text-gray-400 font-mono text-xs">
                    {new Date(alert.created_at).toLocaleString('vi-VN')}
                  </td>
                  <td className="px-4 py-3 text-white">{alert.type?.replace(/_/g, ' ')}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded font-bold
                      ${SEVERITY_BADGE[alert.severity] ?? 'bg-gray-700 text-white'}`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-300 font-mono">{alert.camera_id}</td>
                  <td className="px-4 py-3 text-gray-300">#{alert.global_id}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded
                      ${alert.status === 'acknowledged' ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-300'}`}>
                      {alert.status ?? 'new'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    {alert.status !== 'acknowledged' && (
                      <button
                        onClick={() => acknowledge(alert.id)}
                        className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-green-700 transition-colors"
                      >
                        Xác nhận
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
