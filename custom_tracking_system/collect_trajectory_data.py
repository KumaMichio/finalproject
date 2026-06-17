#!/usr/bin/env python3
"""
CARLA Trajectory Dataset Collector — nâng cấp cho hanoi_district3.

Thu thập trajectory dài hạn trên map thật (Mỗ Lao, Hà Đông) để train:
  - Goal classifier (phân phối xác suất rẽ tại junction)
  - Path generator (snap theo lane centerline)
  - Post-accident physics branch (phase='post_accident')

Output: data/trajectories_hanoi/
  episode_XXXX.csv        — trajectory từng tick
  episode_XXXX_goals.csv  — goal events (xe vào/ra junction)
  run_summary.json        — tổng kết toàn bộ run
  checkpoint.json         — resume point khi crash

CSV trajectory columns:
  t, episode_id, actor_id, class, x, y, vx, vy, speed,
  heading, road_id, lane_id, junction_id, lat, lon, phase

CSV goal columns:
  episode_id, actor_id, class, t_enter, t_exit, junction_id,
  entry_road_id, entry_lane_id, exit_road_id, exit_lane_id,
  turn_angle_deg

Usage (qua dem):
  python collect_trajectory_data.py ^
    --map hanoi_district3 ^
    --episodes 120 --steps-per-episode 1800 ^
    --num-vehicles 20 --scenario-ratio 0.25 ^
    --out data/trajectories_hanoi

Usage (test nhanh):
  python collect_trajectory_data.py --episodes 2 --steps-per-episode 300
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "Map"))

_FIXED_DELTA = 0.1   # 10 Hz — khớp camera_config.yaml
_XODR_PATH   = project_root.parent / "Map" / "hanoi_district3.xodr"

# ──────────────────────────────────────────────────────────
# Blueprint classification helpers
# ──────────────────────────────────────────────────────────
_MOTORCYCLE_IDS = (
    "harley-davidson", "kawasaki", "yamaha", "vespa",
    "crossbike", "century", "gazelle", "diamondback", "bh",
    "low_rider", "ninja", "ysf", "zx125",
)
_BUS_IDS = (
    "volkswagen.t2", "mitsubishi.futura", "carlamotors.carlacola",
)
_TRUCK_IDS = (
    "firetruck", "cybertruck", "ambulance", "sprinter",
)


def classify_blueprint(type_id: str) -> str:
    if type_id.startswith("walker."):
        return "person"
    if type_id.startswith("vehicle."):
        lo = type_id.lower()
        if any(h in lo for h in _MOTORCYCLE_IDS):
            return "motorcycle"
        if any(h in lo for h in _BUS_IDS):
            return "bus"
        if any(h in lo for h in _TRUCK_IDS):
            return "truck"
        return "car"
    return "unknown"


# ──────────────────────────────────────────────────────────
# CARLA connection + map loading
# ──────────────────────────────────────────────────────────

def connect_and_load_map(map_name: str, host: str = "127.0.0.1", port: int = 2000):
    import carla

    client = carla.Client(host, port)
    client.set_timeout(30.0)

    # Retry connection
    for attempt in range(18):
        try:
            sv = client.get_server_version()
            logger.info("Connected to CARLA %s", sv)
            break
        except RuntimeError as e:
            logger.warning("Attempt %d/18: %s", attempt + 1, e)
            time.sleep(5)
    else:
        sys.exit("Cannot connect to CARLA after 90s")

    world = client.get_world()
    current_map = world.get_map().name

    if map_name == "hanoi_district3" and "hanoi" not in current_map.lower():
        logger.info("Loading hanoi_district3 from %s ...", _XODR_PATH)
        if not _XODR_PATH.exists():
            sys.exit(f"XODR not found: {_XODR_PATH}")
        xodr = _XODR_PATH.read_text(encoding="utf-8")
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=0.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
            enable_pedestrian_navigation=False,
        )
        t0 = time.time()
        world = client.generate_opendrive_world(xodr, params, reset_settings=False)
        logger.info("Map loaded in %.1fs", time.time() - t0)
    else:
        logger.info("Using current map: %s", current_map)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = _FIXED_DELTA
    world.apply_settings(settings)
    logger.info("Synchronous mode @ %.0f Hz", 1.0 / _FIXED_DELTA)

    return client, world


# ──────────────────────────────────────────────────────────
# Georeference
# ──────────────────────────────────────────────────────────

def make_georef(map_name: str):
    try:
        from georeference import GeoReference
        if map_name == "hanoi_district3":
            return GeoReference.from_map("hanoi_district3")
    except Exception as e:
        logger.warning("Georeference not available: %s", e)
    return None


def carla_to_latlon(georef, x: float, y: float):
    if georef is None:
        return None, None
    try:
        return georef.carla_to_latlon(x, y)
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────
# Waypoint helper
# ──────────────────────────────────────────────────────────

def get_waypoint_info(carla_map, location):
    """
    Trả về (road_id, lane_id, junction_id) tại location.
    junction_id = -1 nếu không ở trong junction.
    """
    try:
        wp = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=0xFFFF,   # tất cả loại lane
        )
        if wp is None:
            return -1, 0, -1
        jid = wp.get_junction().id if wp.is_junction else -1
        return wp.road_id, wp.lane_id, jid
    except Exception:
        return -1, 0, -1


# ──────────────────────────────────────────────────────────
# Goal event tracker
# ──────────────────────────────────────────────────────────

class GoalTracker:
    """
    Phát hiện goal event: xe vào junction → ra junction → log exit.

    State machine mỗi actor:
      NORMAL → (junction_id > 0) → IN_JUNCTION → (junction_id == -1) → log goal → NORMAL
    """

    def __init__(self):
        self._state = {}   # actor_id -> {'phase': 'normal'|'in_junction',
                           #              'jid': int, 'entry_road': int,
                           #              'entry_lane': int, 't_enter': float,
                           #              'entry_heading': float}

    def update(self, actor_id: int, t: float, road_id: int, lane_id: int,
               junction_id: int, heading_deg: float) -> dict | None:
        """
        Gọi mỗi tick. Trả về goal_event dict khi xe vừa ra khỏi junction,
        hoặc None.
        """
        state = self._state.get(actor_id, {"phase": "normal"})

        if state["phase"] == "normal":
            if junction_id != -1:
                self._state[actor_id] = {
                    "phase":         "in_junction",
                    "jid":           junction_id,
                    "entry_road":    road_id,
                    "entry_lane":    lane_id,
                    "t_enter":       t,
                    "entry_heading": heading_deg,
                }
        elif state["phase"] == "in_junction":
            if junction_id == -1:
                # Vừa ra khỏi junction
                turn = (heading_deg - state["entry_heading"] + 360) % 360
                if turn > 180:
                    turn -= 360   # -180..+180
                event = {
                    "jid":            state["jid"],
                    "entry_road":     state["entry_road"],
                    "entry_lane":     state["entry_lane"],
                    "exit_road":      road_id,
                    "exit_lane":      lane_id,
                    "t_enter":        state["t_enter"],
                    "t_exit":         t,
                    "turn_angle_deg": round(turn, 1),
                }
                self._state[actor_id] = {"phase": "normal"}
                return event
            else:
                # Vẫn trong junction, update junction id phòng nested
                state["jid"] = junction_id
        return None

    def reset_actor(self, actor_id: int):
        self._state.pop(actor_id, None)


# ──────────────────────────────────────────────────────────
# Post-accident tracking
# ──────────────────────────────────────────────────────────

class PostAccidentTracker:
    """
    Đánh dấu actor là 'post_accident' sau khi nhận được collision event.
    Dùng collision sensor CARLA để detect.
    """

    def __init__(self, world):
        self._post_accident: set[int] = set()
        self._sensors: list = []
        self._world = world

    def attach_sensor(self, actor):
        """Gắn collision sensor vào actor, tự cleanup khi actor destroy."""
        import carla
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        sensor = self._world.try_spawn_actor(
            bp,
            carla.Transform(),
            attach_to=actor,
        )
        if sensor:
            actor_id = actor.id

            def _on_collision(event):
                self._post_accident.add(actor_id)
                other_id = event.other_actor.id
                self._post_accident.add(other_id)

            sensor.listen(_on_collision)
            self._sensors.append(sensor)

    def is_post_accident(self, actor_id: int) -> bool:
        return actor_id in self._post_accident

    def cleanup(self):
        for s in self._sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self._sensors.clear()


# ──────────────────────────────────────────────────────────
# Episode collection
# ──────────────────────────────────────────────────────────

_SCENARIO_METHODS = (
    "hit_and_run", "pedestrian_hit", "red_light_crash", "rear_end", "sudden_stop",
)


def collect_episode(client, world, episode_idx: int, args, out_dir: Path,
                    georef, carla_map) -> dict:
    import carla
    from modules.traffic_generator import TrafficGenerator
    from modules.scenario_controller import ScenarioController

    logger.info("=== Episode %d / %d ===", episode_idx + 1, args.episodes)

    rem = max(0.0, 1.0 - args.moto_ratio)
    vehicle_mix = {
        "motorcycle": args.moto_ratio,
        "car":        round(rem * 0.65, 3),
        "truck":      round(rem * 0.20, 3),
        "bus":        round(rem * 0.15, 3),
    }
    traffic = TrafficGenerator(
        world,
        num_vehicles=args.num_vehicles,
        num_pedestrians=args.num_pedestrians,
        vehicle_mix=vehicle_mix,
    )
    traffic.spawn_actors()

    # Collision tracking — chỉ gắn sensor vào scenario vehicles, không gắn all traffic
    # (gắn 19 sensors đồng thời trong sync mode gây tick overload)
    accident_tracker = PostAccidentTracker(world)

    # Scenario (optional)
    scenario = None
    scenario_name = "none"
    if random.random() < args.scenario_ratio:
        scenario = ScenarioController(world, client)
        scenario_name = random.choice(_SCENARIO_METHODS)
        logger.info("  Scenario: %s", scenario_name)
        try:
            getattr(scenario, scenario_name)()
            # Chỉ gắn sensor cho 2-3 actor của scenario
            for actor in scenario._actors:
                if actor.type_id.startswith("vehicle."):
                    accident_tracker.attach_sensor(actor)
        except Exception as e:
            logger.warning("  Scenario failed: %s", e)
            scenario = None
            scenario_name = "none"

    goal_tracker = GoalTracker()
    traj_rows = []
    goal_rows = []

    # Warm-up: cho CARLA ổn định physics + WaypointFollower kiểm soát tốc độ
    # từ tick đầu tiên (tránh xe trôi tự do gây spike tốc độ)
    for _ in range(15):
        try:
            world.tick()
            traffic.update_vehicles()
        except Exception:
            pass

    consecutive_timeouts = 0
    MAX_CONSECUTIVE_TIMEOUTS = 3

    try:
        for step in range(args.steps_per_episode):
            try:
                world.tick()
                consecutive_timeouts = 0
            except Exception as e:
                consecutive_timeouts += 1
                logger.warning("  tick() error at step %d (%d/%d): %s",
                               step, consecutive_timeouts, MAX_CONSECUTIVE_TIMEOUTS, e)
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    logger.error("  CARLA không phản hồi sau %d timeouts — dừng episode sớm",
                                 MAX_CONSECUTIVE_TIMEOUTS)
                    break
                continue

            t = step * _FIXED_DELTA
            traffic.update_vehicles()      # WaypointFollower: giữ xe trên đường
            traffic.update_pedestrians()

            # Collect all actors
            all_actors = []
            for v in traffic.vehicle_list:
                all_actors.append((v, classify_blueprint(v.type_id)))
            for p in traffic.pedestrian_list:
                all_actors.append((p, "person"))
            if scenario is not None:
                for a in scenario._actors:
                    cls = classify_blueprint(a.type_id)
                    if cls != "unknown":
                        all_actors.append((a, cls))

            for actor, cls in all_actors:
                try:
                    loc = actor.get_location()

                    # Bỏ qua xe đang off-road (tránh ghi tọa độ ngoài map)
                    # Chỉ check vehicle, không check pedestrian
                    if cls != "person":
                        snap_wp = carla_map.get_waypoint(
                            loc, project_to_road=True,
                            lane_type=carla.LaneType.Any)
                        if snap_wp is None:
                            continue
                        snap_dist = math.sqrt(
                            (loc.x - snap_wp.transform.location.x) ** 2 +
                            (loc.y - snap_wp.transform.location.y) ** 2)
                        if snap_dist > 5.0:
                            continue   # off-road: không ghi row này

                    vel = actor.get_velocity()
                    rot = actor.get_transform().rotation

                    x, y   = loc.x, loc.y
                    vx, vy = vel.x, vel.y
                    speed  = math.sqrt(vx**2 + vy**2)
                    heading = rot.yaw  # degrees

                    road_id, lane_id, junction_id = get_waypoint_info(carla_map, loc)
                    lat, lon = carla_to_latlon(georef, x, y)

                    phase = "post_accident" if accident_tracker.is_post_accident(actor.id) else "normal"

                    traj_rows.append((
                        round(t, 2), episode_idx, actor.id, cls,
                        round(x, 3), round(y, 3),
                        round(vx, 3), round(vy, 3),
                        round(speed, 3),
                        round(heading, 2),
                        road_id, lane_id, junction_id,
                        round(lat, 7) if lat else "",
                        round(lon, 7) if lon else "",
                        phase,
                    ))

                    # Goal tracking (chỉ cho phương tiện)
                    if cls != "person":
                        event = goal_tracker.update(
                            actor.id, t, road_id, lane_id, junction_id, heading
                        )
                        if event:
                            goal_rows.append((
                                episode_idx, actor.id, cls,
                                round(event["t_enter"], 2),
                                round(event["t_exit"], 2),
                                event["jid"],
                                event["entry_road"], event["entry_lane"],
                                event["exit_road"], event["exit_lane"],
                                event["turn_angle_deg"],
                            ))

                except Exception:
                    pass

    finally:
        accident_tracker.cleanup()
        if scenario is not None:
            try:
                scenario.cleanup()
            except Exception:
                pass
        traffic.cleanup()

    # Lưu trajectory CSV
    traj_path = out_dir / f"episode_{episode_idx:04d}.csv"
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "episode_id", "actor_id", "class",
            "x", "y", "vx", "vy", "speed", "heading",
            "road_id", "lane_id", "junction_id",
            "lat", "lon", "phase",
        ])
        w.writerows(traj_rows)

    # Lưu goal CSV
    goal_path = out_dir / f"episode_{episode_idx:04d}_goals.csv"
    with open(goal_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode_id", "actor_id", "class",
            "t_enter", "t_exit", "junction_id",
            "entry_road_id", "entry_lane_id",
            "exit_road_id", "exit_lane_id",
            "turn_angle_deg",
        ])
        w.writerows(goal_rows)

    n_post = sum(1 for r in traj_rows if r[-1] == "post_accident")
    logger.info(
        "  Episode %d: %d traj rows  %d goals  %d post_accident  scenario=%s",
        episode_idx, len(traj_rows), len(goal_rows), n_post, scenario_name,
    )

    return {
        "episode_id":    episode_idx,
        "scenario":      scenario_name,
        "traj_rows":     len(traj_rows),
        "goal_events":   len(goal_rows),
        "post_accident": n_post,
        "traj_file":     str(traj_path.name),
        "goal_file":     str(goal_path.name),
    }


# ──────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────

def load_checkpoint(out_dir: Path) -> int:
    ckpt = out_dir / "checkpoint.json"
    if ckpt.exists():
        data = json.loads(ckpt.read_text())
        ep = data.get("next_episode", 0)
        logger.info("Resuming from episode %d (checkpoint found)", ep)
        return ep
    return 0


def save_checkpoint(out_dir: Path, next_episode: int):
    ckpt = out_dir / "checkpoint.json"
    ckpt.write_text(json.dumps({"next_episode": next_episode}))


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map",               default="hanoi_district3",
                        help="Map name (default: hanoi_district3)")
    parser.add_argument("--host",              default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=120,
                        help="So episodes (default 120 ~ 6h qua dem)")
    parser.add_argument("--steps-per-episode", type=int, default=1800,
                        help="Steps/episode @ 10Hz -> giay (default 1800 = 3 phut)")
    parser.add_argument("--num-vehicles",  type=int, default=20)
    parser.add_argument("--num-pedestrians", type=int, default=5)
    parser.add_argument("--moto-ratio", type=float, default=0.70,
                        help="Ti le xe may trong traffic (default 0.70 = Hanoi thuc te)")
    parser.add_argument("--scenario-ratio", type=float, default=0.25,
                        help="Ti le episode co scenario tai nan (default 0.25)")
    parser.add_argument("--out",  default="custom_tracking_system/data/trajectories_hanoi",
                        help="Thu muc output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume tu checkpoint neu co (tu dong kich hoat neu checkpoint.json ton tai)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log ra file de xem qua dem
    log_file = out_dir / "collect.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_file)

    # Connect + load map
    client, world = connect_and_load_map(args.map, args.host, args.port)
    carla_map = world.get_map()
    georef    = make_georef(args.map)

    # Resume
    start_ep = load_checkpoint(out_dir) if args.resume or (out_dir / "checkpoint.json").exists() else 0

    episode_summaries = []
    summary_path = out_dir / "run_summary.json"

    # Load existing summary if resuming
    if start_ep > 0 and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        episode_summaries = existing.get("episodes", [])

    t_run_start = time.time()
    logger.info(
        "Starting collection: episodes=%d  steps=%d  vehicles=%d  scenario_ratio=%.2f  start_ep=%d",
        args.episodes, args.steps_per_episode, args.num_vehicles, args.scenario_ratio, start_ep,
    )

    for ep in range(start_ep, args.episodes):
        t0 = time.time()
        try:
            summary = collect_episode(
                client, world, ep, args, out_dir, georef, carla_map
            )
            summary["wall_time_s"] = round(time.time() - t0, 1)
            episode_summaries.append(summary)
        except Exception as e:
            logger.error("Episode %d FAILED: %s", ep, e, exc_info=True)
            episode_summaries.append({
                "episode_id": ep, "error": str(e),
                "wall_time_s": round(time.time() - t0, 1),
            })

        # Checkpoint sau mỗi episode
        save_checkpoint(out_dir, ep + 1)

        # Cập nhật summary JSON
        total_goals = sum(s.get("goal_events", 0) for s in episode_summaries)
        total_rows  = sum(s.get("traj_rows", 0)   for s in episode_summaries)
        total_post  = sum(s.get("post_accident", 0) for s in episode_summaries)
        run_summary = {
            "episodes_done":    ep + 1 - start_ep,
            "episodes_total":   args.episodes,
            "total_traj_rows":  total_rows,
            "total_goal_events": total_goals,
            "total_post_accident_rows": total_post,
            "elapsed_min":      round((time.time() - t_run_start) / 60, 1),
            "args": vars(args),
            "episodes": episode_summaries,
        }
        summary_path.write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        done = ep + 1 - start_ep
        total = args.episodes - start_ep
        elapsed = time.time() - t_run_start
        eta_min = (elapsed / done) * (total - done) / 60 if done else 0
        logger.info(
            "Progress: %d/%d episodes | rows=%d | goals=%d | ETA %.0f min",
            done, total, total_rows, total_goals, eta_min,
        )

    # Final summary
    logger.info("=== COLLECTION DONE ===")
    logger.info("Total traj rows  : %d", sum(s.get("traj_rows", 0) for s in episode_summaries))
    logger.info("Total goal events: %d", sum(s.get("goal_events", 0) for s in episode_summaries))
    logger.info("Total post_acc   : %d", sum(s.get("post_accident", 0) for s in episode_summaries))
    logger.info("Output           : %s", out_dir)

    # Xoa checkpoint khi xong thanh cong
    ckpt = out_dir / "checkpoint.json"
    if ckpt.exists():
        ckpt.unlink()


if __name__ == "__main__":
    main()
