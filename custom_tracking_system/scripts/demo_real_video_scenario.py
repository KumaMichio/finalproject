#!/usr/bin/env python3
"""
demo_real_video_scenario.py — Demo phat hien tai nan + du doan lo trinh tren
video CCTV THAT (khong qua CARLA).

Khac voi demo_accident_scenario.py (CARLA, dung ground-truth actor position):
  - Doc video thi qua FileVideoSource
  - Detect (YOLO11s_vn) + track (ByteTrack) THAT, khong dung GT
  - Pixel -> world qua calibration camera (best-effort, xem calib_pho_hue.json)
  - Goal Classifier dung weights real-world (goal_classifier_real.pkl)
  - Route Predictor chay tren map giao lo thuc (pho_hue_tkc), khong phai CARLA map
  - KHONG co collision sensor -> accident phai duoc danh dau TAY qua API:
        POST http://localhost:5000/api/incidents/{track_id}/mark
    Script se tu phat hien khi state file bi sua tu ben ngoai (manually_marked)
    va tiep tuc chay GoalClassifier + RoutePredictor cho dung track do moi frame.

Usage:
  python scripts/demo_real_video_scenario.py
  python scripts/demo_real_video_scenario.py --video "<path>.mp4" --loop
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from georeference import GeoReference
from route_predictor import RoutePredictor
from modules.detector import ObjectDetector
from modules.tracker import ByteTrackWrapper
from modules.calibration import CameraCalibration
from modules.video_source import FileVideoSource
from modules.goal_classifier import GoalClassifier

MAP_NAME      = 'pho_hue_tkc'
GRAPH_PATH    = ROOT.parent / 'Map' / 'pho_hue_tkc_junction_graph.json'
CALIB_PATH    = ROOT / 'data' / 'calib_pho_hue.json'
DEFAULT_VIDEO = (ROOT / 'data' / 'datasets' / 'traffic_intersection' /
                 '02. Dữ liệu camera giao thông' / 'Phố Huế Trần Khát Chân' /
                 '12h.10.9.22.mp4')
OUT_DIR    = ROOT.parent / 'demo'
STATE_FILE = OUT_DIR / 'incident_state.json'

N_HOPS         = 4
MIN_CONF       = 0.40
HEADING_WINDOW = 5     # world-position samples used to estimate heading via atan2
CAMERA_ID      = 'CAM_PHO_HUE_REAL'
MAX_RANGE_M    = 150.0  # discard ground points beyond this distance from the camera
                         # (pixels near the horizon blow up to absurd world distances
                         # under ground-plane projection -- not a real detection error)


def load_calibration(path: Path, resolution=None) -> CameraCalibration:
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    res = resolution or cfg['resolution']
    return CameraCalibration.from_carla_params(
        position=cfg['position'],
        rotation=cfg['rotation_pitch_yaw_roll'],
        fov=cfg['fov_deg'],
        resolution=res,
        ground_z=cfg.get('ground_z', 0.0),
    )


def write_state(state: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def make_state(phase, accident_track, route, vehicles):
    return {
        'phase': phase,
        'timestamp': time.time(),
        'accident_track': accident_track,
        'predicted_route': route,
        'all_vehicles': vehicles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=str(DEFAULT_VIDEO))
    ap.add_argument('--calib', default=str(CALIB_PATH))
    ap.add_argument('--loop', action='store_true', help='Loop video when it ends')
    ap.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
    ap.add_argument('--max-frames', type=int, default=0, help='Stop after N frames (0=unlimited, for testing)')
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    probe = cv2.VideoCapture(str(video_path))
    fps = probe.get(cv2.CAP_PROP_FPS) or 5.0
    probe.release()
    logger.info("Video FPS: %.2f", fps)

    gr = GeoReference.from_map(MAP_NAME)
    rp = RoutePredictor.from_map(MAP_NAME, georef=gr, graph_path=GRAPH_PATH)
    calib = load_calibration(Path(args.calib))
    with open(args.calib, encoding='utf-8') as f:
        cam_x, cam_y, _ = json.load(f)['position']

    detector = ObjectDetector(model_type=str(ROOT / 'weights' / 'yolo11s_vn.pt'),
                               conf_threshold=args.conf)
    tracker = ByteTrackWrapper(class_map=detector.CLASSES, frame_rate=max(int(round(fps)), 1))
    gc = GoalClassifier(
        str(ROOT / 'weights' / 'goal_classifier_real.pkl'),
        str(ROOT / 'weights' / 'goal_scaler_real.pkl'),
        str(ROOT / 'weights' / 'goal_label_encoder_real.pkl'),
        str(ROOT / 'weights' / 'goal_feature_names_real.json'),
    )

    source = FileVideoSource(str(video_path), CAMERA_ID, loop=args.loop)

    # world-position history per track, for heading estimation: {track_id: [(t,x,y), ...]}
    world_hist: dict[int, list] = {}
    manual_accident_id: int | None = None

    frame_idx = 0
    logger.info("Running real-video demo loop (Ctrl+C to stop)...")
    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                logger.info("Reached --max-frames=%d, stopping.", args.max_frames)
                break
            data = source.get_frame()
            if data is None:
                logger.info("Video ended.")
                break
            frame = cv2.cvtColor(data['frame'], cv2.COLOR_RGB2BGR)
            video_ts = frame_idx / fps
            frame_idx += 1

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

            # Pick up a manual accident mark written by demo/server.py's
            # POST /api/incidents/{id}/mark endpoint (no collision sensor here).
            if manual_accident_id is None:
                ext_state = read_state()
                if ext_state and ext_state.get('accident_track', {}) and \
                        ext_state['accident_track'].get('manually_marked'):
                    manual_accident_id = ext_state['accident_track']['track_id']
                    logger.info("Adopted manually-marked accident track id=%s",
                                manual_accident_id)

            all_veh = []
            accident_track_state = None
            route = []
            phase = 'MONITORING'

            for trk in tracks:
                tid = trk['track_id']
                x1, y1, x2, y2 = trk['box']
                cx, cy_bottom = (x1 + x2) / 2.0, y2  # ground-contact point
                wx, wy = calib.pixel_to_world(cx, cy_bottom)
                if math.hypot(wx - cam_x, wy - cam_y) > MAX_RANGE_M:
                    continue  # near-horizon pixel -> discard implausible ground point
                lat, lon = gr.carla_to_latlon(wx, wy)

                hist = world_hist.setdefault(tid, [])
                hist.append((video_ts, wx, wy))
                if len(hist) > HEADING_WINDOW:
                    hist.pop(0)

                heading_deg = 0.0
                if len(hist) >= 2:
                    (t0, x0, y0), (t1, x1w, y1w) = hist[0], hist[-1]
                    if math.hypot(x1w - x0, y1w - y0) > 0.3:
                        heading_deg = math.degrees(math.atan2(y1w - y0, x1w - x0))

                speed_ms = 0.0
                if len(hist) >= 2:
                    (t0, x0, y0), (t1, x1w, y1w) = hist[0], hist[-1]
                    dt = t1 - t0
                    if dt > 1e-3:
                        speed_ms = math.hypot(x1w - x0, y1w - y0) / dt

                is_accident = (tid == manual_accident_id)
                all_veh.append({
                    'track_id': tid,
                    'x': round(wx, 2), 'y': round(wy, 2),
                    'lat': round(lat, 6), 'lon': round(lon, 6),
                    'speed_ms': round(speed_ms, 2),
                    'heading_deg': round(heading_deg, 1),
                    'is_accident': is_accident,
                })

                if is_accident:
                    direction = 'straight'
                    probs = {'straight': 1.0, 'left': 0.0, 'right': 0.0, 'u_turn': 0.0}
                    conf = 1.0
                    res = gc.update_and_predict(tid, trk['box'], trk['class'], video_ts)
                    if res and res['confidence'] >= MIN_CONF:
                        direction = res['direction']
                        probs = res['probabilities']
                        conf = res['confidence']

                    route = rp.predict(wx, wy, heading_deg, direction=direction,
                                       n_hops=N_HOPS, direction_probs=probs)
                    accident_track_state = {
                        'track_id': tid,
                        'x': round(wx, 2), 'y': round(wy, 2),
                        'lat': round(lat, 6), 'lon': round(lon, 6),
                        'heading_deg': round(heading_deg, 1),
                        'direction': direction,
                        'confidence': round(conf, 3),
                        'probabilities': {k: round(v, 3) for k, v in probs.items()},
                        'manually_marked': True,
                    }
                    phase = 'ROUTE_PREDICTED'

            state = make_state(phase, accident_track_state, route, all_veh)
            write_state(state)

            if frame_idx % 30 == 0:
                logger.info("frame=%d phase=%s tracks=%d route_hops=%d",
                            frame_idx, phase, len(tracks), len(route))

    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
    finally:
        source.release()


if __name__ == '__main__':
    main()
