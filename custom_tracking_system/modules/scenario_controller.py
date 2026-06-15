"""
Scenario Controller Module
Tạo các kịch bản tai nạn có kiểm soát trong CARLA để test hệ thống.
"""

import math
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
        self._collision_cursor: int = 0   # con tro cho get_new_collisions()
        self._last_collision_ts: dict = {}   # {attacker_id: last_report_time} - chong spam

        # cameras_config: dict camera_key -> {position, rotation, camera_id,
        # view_angle, ...} (lấy từ config["cameras"]). Dùng để chọn spawn
        # point nằm trong tầm quan sát của 1 camera, và báo trước camera_id
        # sẽ quan sát được kịch bản.
        self.cameras = list((cameras_config or {}).values())

        # Map mặc định chỉ có rất ít spawn point nằm trong tầm camera (vd.
        # Town10HD_Opt: 155 spawn point nhưng chỉ ~8 nằm trong FOV của 3
        # camera). Bổ sung thêm các "spawn point ảo" lấy từ waypoint trên
        # lane thật, nằm trong tầm quan sát từng camera, để kịch bản tai nạn
        # nhiều khả năng diễn ra trước camera hơn.
        self.spawn_points = list(self.world.get_map().get_spawn_points())
        for cam in self.cameras:
            extra = self._camera_view_spawn_points(cam)
            self.spawn_points.extend(extra)
            logger.debug("Camera %s: +%d spawn point trong tam quan sat",
                          cam.get('camera_id'), len(extra))

        logger.info("ScenarioController initialized (%d cameras, %d spawn points)",
                     len(self.cameras), len(self.spawn_points))

    # ------------------------------------------------------------------
    # Public: chạy kịch bản
    # ------------------------------------------------------------------

    def hit_and_run(self, delay_s: float = 3.0):
        """
        Xe A đâm vào Xe B rồi bỏ trốn.
        Xe A sẽ chạy tiếp (TM ignore vehicles), Xe B dừng lại.
        delay_s: giây chờ trước khi xe A tăng tốc vào xe B.
        """
        spawn_points = self.spawn_points
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
            # Xe A: lái thẳng tới vị trí xe B (đảm bảo va chạm xảy ra gần xe B,
            # tức là trong tầm camera quan sát được pt_b) thay vì để TM tự
            # chọn lộ trình ngẫu nhiên (dễ đi lạc khỏi tầm camera, không bao
            # giờ va chạm).
            attacker.set_autopilot(False, self.tm.get_port())
            logger.info("[HIT_AND_RUN] Attacker đang lao tới nạn nhân...")
            self._pursue(attacker, victim, flee_on_collision=True)

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'attacker': attacker, 'victim': victim, 'camera_id': camera_id}

    def pedestrian_hit(self, delay_s: float = 2.0):
        """Xe tốc độ cao áp sát và đâm người đi bộ."""
        spawn_points = self.spawn_points
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
            # Lái thẳng xe tới vị trí người đi bộ — đảm bảo va chạm xảy ra
            # gần pedestrian (trong tầm camera quan sát được pt) thay vì để
            # TM tự chọn lộ trình ngẫu nhiên.
            vehicle.set_autopilot(False, self.tm.get_port())
            logger.info("[PEDESTRIAN_HIT] Vehicle đang nhắm vào người đi bộ...")
            if pedestrian is not None:
                self._pursue(vehicle, pedestrian)

        threading.Thread(target=_run, daemon=True).start()
        return {'ok': True, 'vehicle': vehicle, 'pedestrian': pedestrian, 'camera_id': camera_id}

    def red_light_crash(self, delay_s: float = 2.0):
        """Hai xe vượt đèn đỏ từ hai hướng đối diện."""
        spawn_points = self.spawn_points
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
        spawn_points = self.spawn_points
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
        spawn_points = self.spawn_points
        if not spawn_points:
            return {'ok': False, 'camera_id': None}

        pt, camera_id = self._pick_spawn_in_view(spawn_points)
        vehicle = self._spawn_vehicle(pt, autopilot=True)
        if not vehicle:
            return {'ok': False, 'camera_id': None}

        logger.info(f"[SUDDEN_STOP] camera quan sát={camera_id}")

        def _run():
            time.sleep(delay_s)
            vehicle.set_autopilot(False, self.tm.get_port())
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

    def get_new_collisions(self) -> list:
        """Trả về các va chạm mới phát sinh kể từ lần gọi trước (và đánh dấu đã đọc)."""
        new = self.collision_log[self._collision_cursor:]
        self._collision_cursor = len(self.collision_log)
        return new

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
        spawn_pts  = self.spawn_points
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
            # KHONG dung controller.ai.walker: controller nay goi RPC
            # 'set_actor_collisions' khong ton tai o server CARLA 0.9.14
            # (client la 0.9.15) -> world.tick() loi RPC nay vinh vien
            # moi frame sau do, lam camera dung hinh. Thay vao do dieu
            # khien pedestrian di bo truc tiep bang WalkerControl.
            dx = goal_pt.location.x - pt.location.x
            dy = goal_pt.location.y - pt.location.y
            dist = max(1e-3, (dx ** 2 + dy ** 2) ** 0.5)
            control = self._carla.WalkerControl(
                direction=self._carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0),
                speed=1.2,
            )
            try:
                ped.apply_control(control)
            except RuntimeError as e:
                logger.warning("Walker apply_control failed: %s", e)
            self._actors.append(ped)
        return ped

    def _pursue(self, chaser, target, max_duration: float = 10.0,
                throttle: float = 1.0, stop_distance: float = 2.5,
                flee_on_collision: bool = False):
        """Lái xe `chaser` (manual control, autopilot=False) trực tiếp về phía
        `target` (vehicle hoặc walker) cho tới khi va chạm (khoảng cách <
        stop_distance), va chạm thực sự được ghi nhận (collision sensor),
        bị kẹt (đứng yên khi đang tăng tốc — vd. đâm vào tường), hoặc hết
        max_duration giây. Dùng để đảm bảo va chạm xảy ra ngay tại vị trí
        target (đã được chọn nằm trong tầm camera) thay vì để TM tự chọn lộ
        trình ngẫu nhiên — có thể đi lạc khỏi tầm camera và không bao giờ va
        chạm.

        flee_on_collision: nếu True và va chạm xảy ra, xe `chaser` sẽ bỏ
        chạy (autopilot, lái ẩu) thay vì dừng lại — dùng cho hit_and_run để
        trajectory predictor dự đoán đường đi tiếp theo của xe bỏ trốn.

        QUAN TRỌNG: luôn dừng xe (throttle=0, brake=1) khi kết thúc — nếu
        không, control cuối (throttle=1.0) sẽ được CARLA giữ nguyên vĩnh
        viễn, khiến xe tiếp tục rà ga vào vật cản sau khi pursuit kết thúc,
        gây quá tải physics engine -> world.tick() timeout -> AI processor
        crash."""
        carla = self._carla
        deadline = time.time() + max_duration
        stuck_since = None
        fled = False
        try:
            while time.time() < deadline:
                if not chaser.is_alive or not target.is_alive:
                    return
                last_col = self._last_collision_ts.get(chaser.id)
                if last_col and time.time() - last_col < 1.0:
                    if flee_on_collision:
                        self._flee(chaser)
                        fled = True
                    return
                t_loc = target.get_transform().location
                c_tf  = chaser.get_transform()
                c_loc = c_tf.location
                dx, dy = t_loc.x - c_loc.x, t_loc.y - c_loc.y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < stop_distance:
                    return
                vel = chaser.get_velocity()
                speed = (vel.x ** 2 + vel.y ** 2) ** 0.5
                if speed < 0.3:
                    stuck_since = stuck_since or time.time()
                    if time.time() - stuck_since > 1.5:
                        return
                else:
                    stuck_since = None
                target_yaw = math.degrees(math.atan2(dy, dx))
                yaw_diff = (target_yaw - c_tf.rotation.yaw + 180) % 360 - 180
                steer = max(-1.0, min(1.0, yaw_diff / 45.0))
                chaser.apply_control(carla.VehicleControl(
                    throttle=throttle, steer=steer, brake=0.0))
                time.sleep(0.1)
        except RuntimeError:
            pass
        finally:
            if not fled:
                try:
                    if chaser.is_alive:
                        chaser.apply_control(carla.VehicleControl(
                            throttle=0.0, steer=0.0, brake=1.0))
                except RuntimeError:
                    pass

    def _flee(self, vehicle):
        """Cho xe bỏ chạy sau va chạm: chuyển sang autopilot, lái ẩu
        (bỏ qua đèn/luật/xe khác, tăng tốc) để rời khỏi hiện trường.
        Track của xe vẫn được AI pipeline theo dõi như bình thường nên
        trajectory predictor sẽ dự đoán đường đi tiếp theo của nó."""
        try:
            port = self.tm.get_port()
            vehicle.set_autopilot(True, port)
            self.tm.ignore_lights_percentage(vehicle, 100)
            self.tm.ignore_signs_percentage(vehicle, 100)
            self.tm.ignore_vehicles_percentage(vehicle, 100)
            self.tm.vehicle_percentage_speed_difference(vehicle, -50)
            logger.info("[FLEE] Xe #%d bỏ chạy khỏi hiện trường", vehicle.id)
        except RuntimeError as e:
            logger.debug("Flee setup warning (non-fatal): %s", e)

    def _attach_collision_sensor(self, vehicle, tag: str = ''):
        bp      = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor  = self.world.spawn_actor(bp, self._carla.Transform(), attach_to=vehicle)

        def _on_collision(event):
            impulse = event.normal_impulse
            strength = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5
            if strength < 100:   # lọc va chạm rất nhẹ
                return
            # Va chạm mạnh (đặc biệt vượt đèn đỏ) có thể bắn collision event
            # hàng nghìn lần/giây khi 2 actor lồng vào nhau -> đánh dấu cooldown
            # NGAY (trước khi gọi _camera_for_point/get_transform) để tránh
            # spam tính toán làm treo CARLA. Chỉ xử lý 1 lần / 8s / attacker.
            now = time.time()
            last = self._last_collision_ts.get(event.actor.id, 0)
            if now - last < 8.0:
                return
            self._last_collision_ts[event.actor.id] = now

            # Chỉ ghi nhận va chạm nếu vị trí va chạm THỰC SỰ nằm trong tầm
            # quan sát của 1 camera lúc này (không dùng camera_id dự đoán
            # lúc spawn — xe có thể đã di chuyển ra ngoài view).
            loc = event.actor.get_transform().location
            visible_camera_id = self._camera_for_point(loc.x, loc.y)
            if visible_camera_id is None:
                return

            entry = {
                'scenario':     tag,
                'camera_id':    visible_camera_id,
                'timestamp':    datetime.now().isoformat(),
                'attacker_id':  event.actor.id,
                'victim_id':    event.other_actor.id,
                'victim_type':  event.other_actor.type_id,
                'strength':     round(strength, 1),
                'scenario_desc': self._describe_scenario(tag),
                'victim_desc':   self._describe_victim_type(event.other_actor.type_id),
                'strength_desc': self._describe_strength(strength),
            }
            self.collision_log.append(entry)
            logger.warning(
                f"[COLLISION] {tag} | attacker={event.actor.id} "
                f"victim={event.other_actor.id} strength={strength:.0f}"
            )

        sensor.listen(_on_collision)
        self._sensors.append(sensor)

    # ------------------------------------------------------------------
    # Mo ta ngon ngu tu nhien (dung cho message hien thi tren dashboard)
    # ------------------------------------------------------------------

    _SCENARIO_DESC = {
        'hit_and_run':    'tai nạn rồi bỏ chạy',
        'pedestrian_hit': 'đâm người đi bộ',
        'red_light_crash': 'vượt đèn đỏ gây va chạm',
        'rear_end':       'đâm từ phía sau',
        'sudden_stop':    'dừng đột ngột gây va chạm',
    }

    def _describe_scenario(self, tag: str) -> str:
        return self._SCENARIO_DESC.get(tag, 'va chạm giao thông')

    def _describe_victim_type(self, type_id: str) -> str:
        if type_id.startswith('vehicle.'):
            return 'một xe khác'
        if type_id.startswith('walker.'):
            return 'người đi bộ'
        if type_id.startswith('traffic.'):
            return 'cột đèn giao thông'
        if type_id.startswith('static.'):
            return 'vật cản trên đường'
        return 'một vật thể'

    def _describe_strength(self, strength: float) -> str:
        if strength > 5000:
            return 'rất mạnh'
        if strength > 1000:
            return 'mạnh'
        return 'nhẹ'

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

    def _camera_view_spawn_points(self, cam, min_range: float = 8.0,
                                   max_range: float = 35.0):
        """Sinh thêm spawn point (carla.Transform) nằm trên lane thật,
        trong tầm quan sát (FOV + [min_range, max_range]) của 1 camera.
        Quét theo vòng tròn quanh camera, dùng get_waypoint(project_to_road)
        để bám đúng lane, dedupe theo (road_id, lane_id, vị trí dọc lane)."""
        import numpy as np
        amap = self.world.get_map()
        cx, cy, _cz = cam.get('position', [0, 0, 0])
        yaw = cam.get('rotation', [0, 0, 0])[1]
        fov = cam.get('view_angle', 90)
        yaw_rad = np.radians(yaw)
        fx, fy = np.cos(yaw_rad), np.sin(yaw_rad)

        seen = set()
        points = []
        for r in np.arange(min_range, max_range + 1e-6, 2.0):
            for ang_deg in range(0, 360, 10):
                a = np.radians(ang_deg)
                x = cx + r * np.cos(a)
                y = cy + r * np.sin(a)

                dx, dy = x - cx, y - cy
                cos_a = (dx * fx + dy * fy) / r
                angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                if angle > fov / 2:
                    continue

                wp = amap.get_waypoint(self._carla.Location(x=x, y=y, z=0.3),
                                        project_to_road=True,
                                        lane_type=self._carla.LaneType.Driving)
                if wp is None:
                    continue

                key = (wp.road_id, wp.lane_id, round(wp.s / 3.0))
                if key in seen:
                    continue
                seen.add(key)

                loc = wp.transform.location
                loc.z += 0.3
                points.append(self._carla.Transform(loc, wp.transform.rotation))
        return points

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
