"""
/ws — WebSocket endpoints cho real-time push.

Clients:
  ws://host:8000/ws/alerts   — nhan alert moi
  ws://host:8000/ws/tracks   — nhan tracking updates
  ws://host:8000/ws/stats    — nhan system stats moi 1s
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Connection manager — quan ly tat ca WebSocket clients
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Quan ly danh sach WebSocket connections theo channel."""

    def __init__(self):
        # {channel_name: set of WebSocket}
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, channel: str, ws: WebSocket):
        await ws.accept()
        if channel not in self._connections:
            self._connections[channel] = set()
        self._connections[channel].add(ws)
        logger.info(f"WS connected: {channel} (total: {len(self._connections[channel])})")

    def disconnect(self, channel: str, ws: WebSocket):
        if channel in self._connections:
            self._connections[channel].discard(ws)
            logger.info(f"WS disconnected: {channel}")

    async def broadcast(self, channel: str, data: dict):
        """Gui message den tat ca clients trong 1 channel."""
        conns = self._connections.get(channel, set()).copy()
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[channel].discard(ws)

    def get_count(self, channel: str) -> int:
        return len(self._connections.get(channel, set()))


# Singleton — import tu cac module khac de broadcast
ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.websocket("/alerts")
async def ws_alerts(ws: WebSocket):
    await ws_manager.connect("alerts", ws)
    try:
        while True:
            # Giu connection song; client co the gui ping/pong
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("alerts", ws)


@router.websocket("/tracks")
async def ws_tracks(ws: WebSocket):
    await ws_manager.connect("tracks", ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("tracks", ws)


@router.websocket("/stats")
async def ws_stats(ws: WebSocket):
    """Push system stats moi 1 giay."""
    await ws_manager.connect("stats", ws)
    try:
        while True:
            # Stats duoc push tu background task (ai_processor)
            # O day chi giu connection song
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("stats", ws)
