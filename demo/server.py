#!/usr/bin/env python3
"""
demo/server.py — Lightweight Flask server cho demo tai nạn.

Endpoints:
  GET  /                        → map_view.html
  GET  /api/state               → incident_state.json (current snapshot)
  POST /api/incidents/{id}/mark → mark vehicle as accident (manually)
  GET  /api/tracks/{id}/routes  → route prediction for a specific track

Run:
  python demo/server.py
  # hoặc với venv:
  ..\custom_tracking_system\venv_tracking\Scripts\python.exe demo\server.py

Then open: http://localhost:5000
"""

import json
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
SYS_ROOT = ROOT.parent / 'custom_tracking_system'
sys.path.insert(0, str(SYS_ROOT))

STATE_FILE = ROOT / 'incident_state.json'

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=str(ROOT), static_url_path='')


def _read_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'phase': 'IDLE', 'timestamp': time.time(),
            'accident_track': None, 'predicted_route': [], 'all_vehicles': []}


# ── Routes ─────────────────────────────────────────────────────

@app.get('/')
def index():
    return send_from_directory(str(ROOT), 'map_view.html')


@app.get('/api/state')
def api_state():
    return jsonify(_read_state())


@app.post('/api/incidents/<int:track_id>/mark')
def api_mark_incident(track_id: int):
    """Manually mark a vehicle as an accident vehicle."""
    state = _read_state()
    # Find the track in all_vehicles
    matched = next((v for v in state.get('all_vehicles', [])
                    if v.get('track_id') == track_id), None)
    if matched is None:
        return jsonify({'error': f'track_id {track_id} not found'}), 404

    state['phase']          = 'ACCIDENT_DETECTED'
    state['accident_track'] = {
        'track_id':    track_id,
        'x':           matched.get('x', 0),
        'y':           matched.get('y', 0),
        'lat':         matched.get('lat'),
        'lon':         matched.get('lon'),
        'heading_deg': matched.get('heading_deg', 0),
        'direction':   'straight',
        'confidence':  1.0,
        'probabilities': {'straight': 1.0, 'left': 0.0, 'right': 0.0, 'u_turn': 0.0},
        'manually_marked': True,
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    return jsonify({'status': 'ok', 'track_id': track_id})


@app.get('/api/tracks/<int:track_id>/routes')
def api_track_routes(track_id: int):
    """Return predicted route for a specific track."""
    state = _read_state()
    at    = state.get('accident_track')
    if at and at.get('track_id') == track_id:
        return jsonify({'route': state.get('predicted_route', [])})
    return jsonify({'route': []})


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()
    print(f"Demo server: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
