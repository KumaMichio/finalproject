import { useEffect, useRef, useState, useCallback } from 'react'

/**
 * WebSocket hook với auto-reconnect.
 * @param {string} path  — ví dụ '/ws/alerts'
 * @param {boolean} enabled
 */
export function useWebSocket(path, enabled = true) {
  const [lastMessage, setLastMessage] = useState(null)
  const [connected, setConnected]     = useState(false)
  const wsRef   = useRef(null)
  const retryRef = useRef(null)

  const connect = useCallback(() => {
    if (!enabled) return
    const url = `ws://${window.location.host}${path}`
    const ws  = new WebSocket(url)

    ws.onopen    = () => { setConnected(true); clearTimeout(retryRef.current) }
    ws.onmessage = (e) => { setLastMessage(e) }
    ws.onclose   = () => {
      setConnected(false)
      retryRef.current = setTimeout(connect, 3000)
    }
    ws.onerror   = () => ws.close()

    wsRef.current = ws
  }, [path, enabled])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { lastMessage, connected }
}
