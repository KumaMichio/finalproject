"""
Scenario Controller Module
Tạo các kịch bản tai nạn có kiểm soát trong CARLA để test hệ thống.
"""

import time
import random
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ScenarioController:
    """
    Script kịch bản tai nạn trong CARLA.

    Các kịch bản có sẵn:
      - hit_and_run       : Xe A đâm Xe B rồi bỏ trốn
      - pedestrian_hit    : Xe đâm người đi bộ
      - red_light_crash   : Hai xe vượt đèn đỏ từ hai hướng đâm nhau
      - rear_end          : Xe đâm từ phía sau
      - sudden_stop       : Xe dừng đột ngột giữa đường
    """

    def __init__(self, world, client, cameras_config: dict = None):
        import carla
        self._carla = carla
        self.world  = world
        self.client = client
        self.tm     = client.get_trafficmanager(8001)
        self.tm.set_synchronous_mode(True)

        self._sensors:  list = []   # collision sensors
        self._actors:   list = []   # actors do scenario tạo ra
        self.collision_log: list = []

        # cameras_config: dict camera_key -> {position, rotation, camera_id,
        # view_angle, ...} (lấy từ config["cameras"]). Dùng để chọn spawn
        # point nằm trong tầm quan sát của 1 camera, và báo trước camera_id
        # sẽ quan sát được kịch bản.
        self.cameras = list((cameras_config or {}).values())

        logger.info("ScenarioController initialized (%d cameras)", len(self.cameras))

    # ------------------------------------------------------------------
    # Public: chạy kịch bản
    # ------------------------------------------------------------------

    def hit_and_run(self, delay_s: float = 3.0):
        """
        Xe A đâm vào Xe B rồi bỏ trốn.
        Xe A sẽ chạy tiếp (TM ignore vehicles), Xe B dừng lại.
        delay_s: giây chờ trước khi xe A tăng tốc vào xe B.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            logger.error("Không đủ spawn points cho kịch bản hit_and_run")
            return {'ok': False, 'camera_id': None}

        # Chọn 2 điểm gần nhau, cùng hướng — ưu tiên điểm nằm trong tầm camera
        pt_b, camera_id = self._pick_spawn_in_view(spawn_points)
        pt_a = self._find_nearby_spawn(spawn_points, pt_b, distance=20)

        victim   = self._spawn_vehicle(pt_b, autopilot=True)
        attacker = self._spawn_vehicle(pt_a, autopilot=False)

        if not victim or not attacker:
            return {'ok': False, 'camera_id': None}

        self._attach_collision_sensor(attacker, tag='hit_and_run')

        logger.info(f"[HIT_AND_RUN] Victim={victim.id}, Attacker={attacker.id} — "
                    f"va chạm sau {delay_s}s, camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            # Xe A: đâm thẳng về phía trước
            attacker.set_autopilot(True)
            self.tm.vehicle_percentage_speed_difference(attacker, -80)  # nhanh hơn 80%
            self.tm.distance_to_leading_vehicle(attacker, 0)
            self.tm.ignore_vehicles_percentage(attacker, 100)
            logger.info("[HIT_AND_RUN] Attacker đang tiến tới nạn nhân...")

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'attacker': attacker, 'victim': victim, 'camera_id': camera_id}

    def pedestrian_hit(self, delay_s: float = 2.0):
        """Xe tốc độ cao áp sát và đâm người đi bộ."""
        spawn_points = self.world.get_map().get_spawn_points()
        bp_library   = self.world.get_blueprint_library()

        if not spawn_points:
            return {'ok': False, 'camera_id': None}

        pt, camera_id = self._pick_spawn_in_view(spawn_points)
        vehicle    = self._spawn_vehicle(pt, autopilot=True)
        pedestrian = self._spawn_pedestrian(bp_library, near=pt)

        if not vehicle:
            return {'ok': False, 'camera_id': None}

        self._attach_collision_sensor(vehicle, tag='pedestrian_hit')

        logger.info(f"[PEDESTRIAN_HIT] Vehicle={vehicle.id} — camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            self.tm.ignore_walkers_percentage(vehicle, 100)
            self.tm.vehicle_percentage_speed_difference(vehicle, -40)
            logger.info("[PEDESTRIAN_HIT] Vehicle đang nhắm vào người đi bộ...")

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'vehicle': vehicle, 'pedestrian': pedestrian, 'camera_id': camera_id}

    def red_light_crash(self, delay_s: float = 2.0):
        """Hai xe vượt đèn đỏ từ hai hướng đối diện."""
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            return {'ok': False, 'camera_id': None}

        pt_a, camera_id = self._pick_spawn_in_view(spawn_points)
        pt_b = self._find_opposite_spawn(spawn_points, pt_a)

        v_a = self._spawn_vehicle(pt_a, autopilot=True)
        v_b = self._spawn_vehicle(pt_b, autopilot=True)

        if not v_a or not v_b:
            return {'ok': False, 'camera_id': None}

        self._attach_collision_sensor(v_a, tag='red_light_crash')
        self._attach_collision_sensor(v_b, tag='red_light_crash')

        logger.info(f"[RED_LIGHT_CRASH] camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            for v in (v_a, v_b):
                self.tm.ignore_lights_percentage(v, 100)
                self.tm.vehicle_percentage_speed_difference(v, -30)
                self.tm.ignore_vehicles_percentage(v, 100)
            logger.info("[RED_LIGHT_CRASH] Hai xe đang vượt đèn đỏ...")

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'vehicle_a': v_a, 'vehicle_b': v_b, 'camera_id': camera_id}

    def rear_end(self, delay_s: float = 2.0):
        """Xe đâm từ phía sau xe đang chạy chậm."""
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            return {'ok': False, 'camera_id': None}

        pt_front, camera_id = self._pick_spawn_in_view(spawn_points)
        pt_rear  = self._find_nearby_spawn(spawn_points, pt_front, distance=15)

        front_car = self._spawn_vehicle(pt_front, autopilot=True)
        rear_car  = self._spawn_vehicle(pt_rear,  autopilot=True)

        if not front_car or not rear_car:
            return {'ok': False, 'camera_id': None}

        self._attach_collision_sensor(rear_car, tag='rear_end')

        logger.info(f"[REAR_END] camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            # Xe trước chạy chậm
            self.tm.vehicle_percentage_speed_difference(front_car, 50)
            # Xe sau tăng tốc, không giữ khoảng cách
            self.tm.vehicle_percentage_speed_difference(rear_car, -60)
            self.tm.distance_to_leading_vehicle(rear_car, 0)
            logger.info("[REAR_END] Xe phía sau đang đâm vào xe phía trước...")

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'rear_car': rear_car, 'front_car': front_car, 'camera_id': camera_id}

    def sudden_stop(self, delay_s: float = 3.0):
        """Xe dừng đột ngột giữa đường."""
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return {'ok': False, 'camera_id': None}

        pt, camera_id = self._pick_spawn_in_view(spawn_points)
        vehicle = self._spawn_vehicle(pt, autopilot=True)
        if not vehicle:
            return {'ok': False, 'camera_id': None}

        logger.info(f"[SUDDEN_STOP] camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            vehicle.set_autopilot(False)
            vehicle.apply_control(self._carla.VehicleControl(
                throttle=0.0, brake=1.0, steer=0.0
            ))
            logger.info(f"[SUDDEN_STOP] Xe #{vehicle.id} dừng đột ngột")

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'vehicle': vehicle, 'camera_id': camera_id}

    # ------------------------------------------------------------------
    # Info & Cleanup
    # ------------------------------------------------------------------

    def get_collision_log(self) -> list:
        """Trả về log va chạm đã xảy ra."""
        return self.collision_log.copy()

    def cleanup(self):
        """Xoá tất cả actors và sensors do scenario tạo ra."""
        for sensor in self._sensors:
            try: sensor.destroy()
            except Exception: pass

        for actor in self._actors:
            try: actor.destroy()
            except Exception: pass

        self._sensors.clear()
        self._actors.clear()
        logger.info("ScenarioController cleanup done")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_vehicle(self, transform, autopilot: bool = True):
        bp_lib = self.world.get_blueprint_library()
        bps    = bp_lib.filter('vehicle.audi.*') or bp_lib.filter('vehicle.*')
        bp     = random.choice(list(bps))
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))

        actor = self.world.try_spawn_actor(bp, transform)
        if actor:
            if autopilot:
                actor.set_autopilot(True, self.tm.get_port())
            self._actors.append(actor)
            logger.debug(f"Spawned vehicle id={actor.id} at {transform.location}")
        else:
            logger.warning("Failed to spawn vehicle")
        return actor

    def _spawn_pedestrian(self, bp_library, near=None):
        """Spawn 1 pedestrian. Nếu `near` (carla.Transform) được cung cấp,
        ưu tiên spawn gần vị trí đó (trong tầm camera) và cho đi bộ về một
        điểm gần đó, để pedestrian + vehicle cùng nằm trong khung hình."""
        walker_bps = bp_library.filter('walker.pedestrian.*')
        ctrl_bp    = bp_library.find('controller.ai.walker')
        spawn_pts  = self.world.get_map().get_spawn_points()
        if not walker_bps or not spawn_pts:
            return None

        bp = random.choice(list(walker_bps))
        if near is not None:
            pt = self._find_nearby_spawn(spawn_pts, near, distance=15)
            goal_pt = near
        else:
            pt = random.choice(spawn_pts)
            goal_pt = random.choice(spawn_pts)

        ped = self.world.try_spawn_actor(bp, pt)
        if ped:
            self.world.tick()  # cần tick trước khi spawn controller
            ctrl = self.world.spawn_actor(ctrl_bp, self._carla.Transform(), attach_to=ped)
            if ctrl:
                self._actors.append(ctrl)
                try:
                    ctrl.start()
                    ctrl.go_to_location(goal_pt.location)
                    ctrl.set_max_speed(1.2)
                except RuntimeError as e:
                    # walker AI controller co the loi rpc do version mismatch
                    # CARLA client/server — pedestrian van duoc spawn, chi
                    # khong di bo (dung yen), khong lam fail ca kich ban.
                    logger.warning("Walker AI controller start failed: %s", e)
            self._actors.append(ped)
        return ped

    def _attach_collision_sensor(self, vehicle, tag: str = ''):
        bp      = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor  = self.world.spawn_actor(bp, self._carla.Transform(), attach_to=vehicle)

        def _on_collision(event):
            impulse = event.normal_impulse
            strength = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5
            if strength < 100:   # lọc va chạm rất nhẹ
                return
            entry = {
                'scenario':    tag,
                'timestamp':   datetime.now().isoformat(),
                'attacker_id': event.actor.id,
                'victim_id':   event.other_actor.id,
                'victim_type': event.other_actor.type_id,
                'strength':    round(strength, 1),
            }
            self.collision_log.append(entry)
            logger.warning(
                f"[COLLISION] {tag} | attacker={event.actor.id} "
                f"victim={event.other_actor.id} strength={strength:.0f}"
            )

        sensor.listen(_on_collision)
        self._sensors.append(sensor)

    def _find_nearby_spawn(self, spawn_points, reference, distance: float = 20):
        """Tìm spawn point gần reference trong khoảng distance (m), không trùng."""
        import numpy as np
        ref_loc = reference.location
        candidates = [
            p for p in spawn_points
            if p != reference and
            np.linalg.norm([p.location.x - ref_loc.x,
                            p.location.y - ref_loc.y]) < distance * 2
        ]
        return random.choice(candidates) if candidates else random.choice(spawn_points)

    def _find_opposite_spawn(self, spawn_points, reference):
        """Tìm spawn point đối diện hướng (yaw ~180° khác)."""
        import numpy as np
        ref_yaw = reference.rotation.yaw
        opposite = [
            p for p in spawn_points
            if p != reference and
            abs(abs(p.rotation.yaw - ref_yaw) - 180) < 30
        ]
        return random.choice(opposite) if opposite else random.choice(spawn_points)

    # ------------------------------------------------------------------
    # Camera-aware spawn selection
    # ------------------------------------------------------------------

    def _camera_for_point(self, x: float, y: float, max_range: float = 40.0):
        """Trả về camera_id quan sát được điểm (x, y) trong CARLA world,
        hoặc None nếu không camera nào phủ tới. Dùng vị trí + yaw + FOV
        (view_angle) của từng camera trong camera_config.yaml — coi camera
        nhìn ngang (pitch=0), hướng nhìn = (cos(yaw), sin(yaw))."""
        import numpy as np
        best_cam, best_dist = None, None
        for cam in self.cameras:
            cx, cy, _cz = cam.get('position', [0, 0, 0])
            yaw = cam.get('rotation', [0, 0, 0])[1]
            fov = cam.get('view_angle', 90)

            dx, dy = x - cx, y - cy
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < 1e-6 or dist > max_range:
                continue

            yaw_rad = np.radians(yaw)
            fx, fy = np.cos(yaw_rad), np.sin(yaw_rad)
            cos_angle = (dx * fx + dy * fy) / dist
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            if angle <= fov / 2 and (best_dist is None or dist < best_dist):
                best_cam, best_dist = cam.get('camera_id'), dist
        return best_cam

    def _pick_spawn_in_view(self, spawn_points):
        """Chọn 1 spawn point nằm trong tầm quan sát của 1 camera nếu có,
        để kịch bản tai nạn diễn ra trước camera (thấy được trên dashboard).
        Trả về (spawn_point, camera_id) — camera_id là None nếu không spawn
        point nào nằm trong tầm camera (fallback random)."""
        if self.cameras:
            covered = []
            for p in spawn_points:
                cam_id = self._camera_for_point(p.location.x, p.location.y)
                if cam_id:
                    covered.append((p, cam_id))
            if covered:
                return random.choice(covered)
            logger.warning("Không tìm thấy spawn point nào trong tầm camera nào — "
                           "chọn ngẫu nhiên (có thể không quan sát được)")
        return random.choice(spawn_points), None
