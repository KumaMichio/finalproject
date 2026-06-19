#!/usr/bin/env python3
"""
demo_accident_scenario.py — Kịch bản demo phát hiện tai nạn + dự đoán lộ trình.

Flow:
  1. Load hanoi_district3.xodr vào CARLA
  2. Spawn N vehicles với TrafficManager autopilot
  3. Gắn collision sensor vào tất cả xe
  4. Force collision: disable autopilot xe target → đâm vào xe khác
  5. Khi nhận collision event → mark accident vehicle
  6. Chạy GoalClassifier + RoutePredictor trên xe đó
  7. Xuất predicted route → demo/incident_state.json (đọc bởi map_view.html)
  8. Visualize trên CARLA spectator camera

Usage:
  python scripts/demo_accident_scenario.py --host localhost --port 2000
  python scripts/demo_accident_scenario.py --dry-run    # test không cần CARLA
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

from georeference import GeoReference
from route_predictor import RoutePredictor

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
XODR_PATH    = ROOT.parent / 'Map' / 'hanoi_district3.xodr'
OUT_DIR      = ROOT.parent / 'demo'
STATE_FILE   = OUT_DIR / 'incident_state.json'
N_VEHICLES   = 12
N_HOPS       = 4          # how many junctions to predict ahead
MIN_CONF     = 0.40       # min GoalClassifier confidence to use predicted direction
SYNC_DELTA   = 0.05       # seconds per tick (20 Hz)

# Junction spawn region (CARLA coords, central area of map)
SPAWN_CENTER = (250.0, -300.0)
SPAWN_RADIUS = 200.0

# Camera setup — 3 key junctions (will be set as spectator attachment points)
CAMERA_POSITIONS = [
    {'label': 'cam0', 'x': 520.0, 'y': -463.0, 'z': 25.0, 'pitch': -45.0, 'yaw': 0.0},
    {'label': 'cam1', 'x': 32.0,  'y': -280.0, 'z': 25.0, 'pitch': -45.0, 'yaw': 90.0},
    {'label': 'cam2', 'x': 250.0, 'y': -300.0, 'z': 25.0, 'pitch': -60.0, 'yaw': 45.0},
]


# ──────────────────────────────────────────────────────────────
# State output helpers
# ──────────────────────────────────────────────────────────────

def write_state(state: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def make_state(phase: str, accident_track: dict | None,
               route: list[dict], vehicles: list[dict]) -> dict:
    return {
        'phase':          phase,
        'timestamp':      time.time(),
        'accident_track': accident_track,
        'predicted_route': route,
        'all_vehicles':   vehicles,
    }


# ──────────────────────────────────────────────────────────────
# DRY-RUN mode (test without CARLA)
# ──────────────────────────────────────────────────────────────

def dry_run():
    logger.info("=== DRY-RUN mode (no CARLA) ===")
    gr = GeoReference.from_map('hanoi_district3')
    rp = RoutePredictor.from_map('hanoi_district3', georef=gr)

    # Simulate accident vehicle at junction 3 approach, heading east
    acc_x, acc_y = 490.0, -455.0
    heading_deg  = -20.0   # roughly east-south-east
    direction    = 'right'
    probs        = {'straight': 0.30, 'right': 0.55, 'left': 0.12, 'u_turn': 0.03}

    lat, lon = gr.carla_to_latlon(acc_x, acc_y)
    route    = rp.predict(acc_x, acc_y, heading_deg, direction=direction,
                          n_hops=N_HOPS, direction_probs=probs)

    accident_track = {
        'track_id':    42,
        'x': acc_x, 'y': acc_y,
        'lat': lat,  'lon': lon,
        'heading_deg': heading_deg,
        'direction':   direction,
        'confidence':  probs[direction],
        'probabilities': probs,
    }

    vehicles = [
        {'track_id': 42, 'x': acc_x, 'y': acc_y,
         'lat': lat, 'lon': lon, 'is_accident': True},
        {'track_id': 10, 'x': 400.0, 'y': -420.0,
         'lat': gr.carla_to_latlon(400.0, -420.0)[0],
         'lon': gr.carla_to_latlon(400.0, -420.0)[1],
         'is_accident': False},
    ]

    state = make_state('ROUTE_PREDICTED', accident_track, route, vehicles)
    write_state(state)

    logger.info("Accident vehicle: track #42 at (%.1f, %.1f) → lat=%.5f lon=%.5f",
                acc_x, acc_y, lat, lon)
    logger.info("Predicted direction: %s (conf=%.0f%%)", direction, probs[direction]*100)
    logger.info("Route (%d hops):", len(route))
    for wp in route:
        lat_s = f"{wp['lat']:.5f},{wp['lon']:.5f}" if wp['lat'] else "n/a"
        logger.info("  hop%d road=%-5s (%.1f, %.1f)  hdg=%.1f°  dir=%s  latlon=%s",
                    wp['hop'], wp['road_id'], wp['x'], wp['y'],
                    wp['heading_deg'], wp['direction_used'], lat_s)
    logger.info("State written → %s", STATE_FILE)


# ──────────────────────────────────────────────────────────────
# CARLA mode
# ──────────────────────────────────────────────────────────────

def run_carla(host: str, port: int, seed: int):
    try:
        import carla
    except ImportError:
        logger.error("carla module not found. Use --dry-run for testing without CARLA.")
        sys.exit(1)

    gr = GeoReference.from_map('hanoi_district3')
    rp = RoutePredictor.from_map('hanoi_district3', georef=gr)

    # ── Connect ────────────────────────────────────────────
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    logger.info("Connected to CARLA %s", client.get_server_version())

    # ── Load hanoi_district3 ───────────────────────────────
    xodr_content = XODR_PATH.read_text()
    logger.info("Loading hanoi_district3.xodr (%d bytes)...", len(xodr_content))
    world = client.generate_opendrive_world(
        xodr_content,
        carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=500.0,
            wall_height=0.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
        ),
    )
    logger.info("World loaded: %s", world.get_map().name)

    # ── Sync mode ─────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SYNC_DELTA
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_random_device_seed(seed)
    tm.global_percentage_speed_difference(-20)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("No spawn points found in hanoi_district3 map!")
        _cleanup(world, settings, [])
        return

    rng = __import__('random')
    rng.seed(seed)
    rng.shuffle(spawn_points)

    vehicles  = []
    sensors   = []
    collision_events = {}  # actor_id -> event

    try:
        # ── Spawn vehicles ──────────────────────────────────
        n_spawned = 0
        car_bps = blueprint_library.filter('vehicle.toyota.*') + \
                  blueprint_library.filter('vehicle.audi.*')
        if not car_bps:
            car_bps = blueprint_library.filter('vehicle.*')

        for i, sp in enumerate(spawn_points[:N_VEHICLES * 2]):
            bp = rng.choice(list(car_bps))
            bp.set_attribute('role_name', f'demo_{i}')
            actor = world.try_spawn_actor(bp, sp)
            if actor is None:
                continue
            actor.set_autopilot(True, tm.get_port())
            vehicles.append(actor)
            n_spawned += 1
            if n_spawned >= N_VEHICLES:
                break

        logger.info("Spawned %d vehicles", len(vehicles))
        if len(vehicles) < 2:
            logger.error("Not enough vehicles spawned")
            return

        # ── Pick target (accident) vehicle ──────────────────
        target_vehicle = vehicles[0]
        logger.info("Target (accident) vehicle: id=%d", target_vehicle.id)

        # ── Attach collision sensors ────────────────────────
        col_bp = blueprint_library.find('sensor.other.collision')
        for v in vehicles:
            sensor = world.try_spawn_actor(
                col_bp, carla.Transform(), attach_to=v)
            if sensor:
                vid = v.id

                def _on_col(event, vid=vid):
                    collision_events[vid] = {
                        'actor_id':    vid,
                        'other_id':    event.other_actor.id,
                        'impulse_kgm': event.normal_impulse.length(),
                        'timestamp':   event.timestamp,
                    }

                sensor.listen(_on_col)
                sensors.append(sensor)

        # ── GoalClassifier (lazy import) ─────────────────────
        try:
            from modules.goal_classifier import GoalClassifier
            gc = GoalClassifier(
                str(ROOT / 'weights' / 'goal_classifier_real.pkl'),
                str(ROOT / 'weights' / 'goal_scaler_real.pkl'),
                str(ROOT / 'weights' / 'goal_label_encoder_real.pkl'),
                str(ROOT / 'weights' / 'goal_feature_names_real.json'),
            )
        except FileNotFoundError:
            logger.warning("GoalClassifier weights not found — direction will be 'straight'")
            gc = None

        # ── Warm-up ticks ────────────────────────────────────
        WARMUP_TICKS = 80
        logger.info("Warm-up: %d ticks...", WARMUP_TICKS)
        for _ in range(WARMUP_TICKS):
            world.tick()

        # ── Force collision ───────────────────────────────────
        logger.info("Forcing collision on vehicle id=%d...", target_vehicle.id)
        target_vehicle.set_autopilot(False, tm.get_port())
        tgt_tf = target_vehicle.get_transform()
        fwd    = tgt_tf.get_forward_vector()
        target_vehicle.apply_control(carla.VehicleControl(
            throttle=1.0, steer=0.0, brake=0.0))

        FORCE_TICKS = 30
        for _ in range(FORCE_TICKS):
            world.tick()
            if target_vehicle.id in collision_events:
                break

        target_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        # ── Wait for collision event ──────────────────────────
        logger.info("Waiting for collision event...")
        for _ in range(60):
            world.tick()
            if target_vehicle.id in collision_events:
                break

        phase = 'MONITORING'
        accident_track_state = None

        if target_vehicle.id in collision_events:
            ev = collision_events[target_vehicle.id]
            logger.info("COLLISION DETECTED: id=%d impulse=%.1f",
                        ev['actor_id'], ev['impulse_kgm'])
            phase = 'ACCIDENT_DETECTED'

            # Re-enable autopilot so vehicle keeps moving
            target_vehicle.set_autopilot(True, tm.get_port())

        # ── Tracking + Route prediction loop ─────────────────
        logger.info("Running route prediction loop (press Ctrl+C to stop)...")
        frame = 0
        try:
            while True:
                world.tick()
                frame += 1
                now_ts = time.time()

                # Collect all vehicle positions
                all_veh = []
                for v in vehicles:
                    if not v.is_alive:
                        continue
                    tf  = v.get_transform()
                    vel = v.get_velocity()
                    spd = math.sqrt(vel.x**2 + vel.y**2)
                    vx, vy = tf.location.x, tf.location.y
                    lat, lon = gr.carla_to_latlon(vx, vy)
                    all_veh.append({
                        'track_id':   v.id,
                        'x': round(vx, 2), 'y': round(vy, 2),
                        'lat': round(lat, 6), 'lon': round(lon, 6),
                        'speed_ms': round(spd, 2),
                        'heading_deg': round(tf.rotation.yaw, 1),
                        'is_accident': v.id in collision_events,
                    })

                # Run GoalClassifier on accident vehicle
                route = []
                if target_vehicle.is_alive and target_vehicle.id in collision_events:
                    tf  = target_vehicle.get_transform()
                    vx, vy = tf.location.x, tf.location.y
                    hdg    = tf.rotation.yaw
                    lat, lon = gr.carla_to_latlon(vx, vy)

                    direction = 'straight'
                    probs     = {'straight': 1.0, 'left': 0.0, 'right': 0.0, 'u_turn': 0.0}
                    conf      = 1.0

                    if gc is not None:
                        box = [vx - 2, vy - 1, vx + 2, vy + 1]
                        res = gc.update_and_predict(
                            target_vehicle.id, box, 'car', now_ts)
                        if res and res['confidence'] >= MIN_CONF:
                            direction = res['direction']
                            probs     = res['probabilities']
                            conf      = res['confidence']

                    route = rp.predict(vx, vy, hdg, direction=direction,
                                       n_hops=N_HOPS, direction_probs=probs)

                    accident_track_state = {
                        'track_id':    target_vehicle.id,
                        'x': round(vx, 2), 'y': round(vy, 2),
                        'lat': round(lat, 6), 'lon': round(lon, 6),
                        'heading_deg': round(hdg, 1),
                        'direction':   direction,
                        'confidence':  round(conf, 3),
                        'probabilities': {k: round(v, 3) for k, v in probs.items()},
                    }
                    phase = 'ROUTE_PREDICTED'

                state = make_state(phase, accident_track_state, route, all_veh)
                write_state(state)

                if frame % 20 == 0:
                    logger.info("frame=%d  phase=%s  route_hops=%d",
                                frame, phase, len(route))

        except KeyboardInterrupt:
            logger.info("Demo stopped by user.")

    finally:
        _cleanup(world, settings, vehicles + sensors)


def _cleanup(world, settings, actors):
    logger.info("Cleaning up %d actors...", len(actors))
    for a in actors:
        try:
            if hasattr(a, 'stop'):
                a.stop()
            a.destroy()
        except Exception:
            pass
    settings.synchronous_mode = False
    world.apply_settings(settings)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host',    default='localhost')
    ap.add_argument('--port',    type=int, default=2000)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--dry-run', action='store_true',
                    help='Test without CARLA — uses hardcoded vehicle position')
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
    else:
        run_carla(args.host, args.port, args.seed)
