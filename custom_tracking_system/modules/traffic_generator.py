"""
Traffic Generator Module
Spawns and manages vehicles and pedestrians in CARLA simulation.

Dùng WaypointFollower thay cho TM autopilot vì custom OpenDRIVE map
không đảm bảo TM route đúng. WaypointFollower mô phỏng hành vi lái xe
thực tế: giảm tốc khi rẽ, tốc độ theo loại xe, tính cách lái ngẫu nhiên.
"""

import carla
import math
import random
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _dist2d(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _classify_type_id(type_id: str) -> str:
    lo = type_id.lower()
    if any(k in lo for k in ("harley", "kawasaki", "yamaha", "vespa",
                              "ninja", "low_rider", "crossbike", "century",
                              "gazelle", "diamondback", "bh", "ysf", "zx125")):
        return "motorcycle"
    if any(k in lo for k in ("volkswagen.t2", "mitsubishi.futura", "carlamotors")):
        return "bus"
    if any(k in lo for k in ("firetruck", "cybertruck", "ambulance", "sprinter")):
        return "truck"
    if lo.startswith("vehicle."):
        return "car"
    return "car"


# ──────────────────────────────────────────────────────────
# Waypoint Follower — hành vi lái xe thực tế
# ──────────────────────────────────────────────────────────

class WaypointFollower:
    """
    Pure-pursuit waypoint follower với mô phỏng hành vi lái xe thực tế:
    - Tốc độ tối đa theo loại xe (car 60, moto 50, bus/truck 40 km/h)
    - Giảm tốc khi rẽ/quay đầu theo độ cong của đường phía trước
    - Mỗi xe có "tính cách lái" ngẫu nhiên (nhanh/chậm hơn một chút)
    - Gia tốc/phanh mượt, không tức thì
    """

    # Tốc độ tối đa theo class (m/s)
    _SPEED_LIMITS = {
        "car":        16.7,   # 60 km/h — nội thành
        "motorcycle": 13.9,   # 50 km/h
        "bus":        11.1,   # 40 km/h
        "truck":       9.7,   # 35 km/h
    }
    _DEFAULT_SPEED = 13.9    # fallback

    LOOKAHEAD_DIST   = 4.0   # m — lookahead của pure-pursuit
    ROUTE_LEN        = 400   # waypoints xây trước
    CURVE_LOOKAHEAD  = 10    # waypoints nhìn trước để tính độ cong
    OFFROAD_BRAKE_TICKS = 0  # số tick đang off-road (để brake cứng)

    def __init__(self, carla_map):
        self._map          = carla_map
        self._routes       = {}   # actor_id -> list[carla.Location]
        self._speed_limit  = {}   # actor_id -> tốc độ tối đa (m/s)
        self._personality  = {}   # actor_id -> hệ số tốc độ ưa thích (0.6–1.0)
        self._target_speed = {}   # actor_id -> target speed hiện tại (m/s, smoothed)
        self._speed_jitter = {}   # actor_id -> (timer, current_jitter)
        self._offroad_ticks = {}  # actor_id -> số tick liên tiếp off-road

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def register(self, vehicle):
        """Đăng ký xe mới: xác định class, speed limit, tính cách lái."""
        cls   = _classify_type_id(vehicle.type_id)
        limit = self._SPEED_LIMITS.get(cls, self._DEFAULT_SPEED)

        self._speed_limit[vehicle.id] = limit
        # Tính cách lái: 60–95% speed limit, phân phối chuẩn lệch về chậm
        personality = random.triangular(0.60, 0.95, 0.75)
        self._personality[vehicle.id] = personality
        self._target_speed[vehicle.id] = limit * personality
        self._speed_jitter[vehicle.id] = (0, 0.0)

    def assign_route(self, vehicle, from_spawn=False):
        """Xây route từ vị trí hiện tại. Dead end → backtrack bằng previous().
        from_spawn=True: spawn point ngẫu nhiên thay vì vị trí hiện tại (khi off-road).
        """
        if from_spawn:
            spawn_points = self._map.get_spawn_points()
            if not spawn_points:
                self._routes[vehicle.id] = []
                return
            # Tìm spawn point gần nhất vẫn còn trên đường
            loc  = vehicle.get_location()
            spawn = min(spawn_points, key=lambda sp: _dist2d(loc, sp.location))
            wp = self._map.get_waypoint(spawn.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        else:
            loc = vehicle.get_location()
            wp  = self._map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)

        if wp is None:
            self._routes[vehicle.id] = []
            return

        route = [wp]
        for _ in range(self.ROUTE_LEN):
            nexts = route[-1].next(2.0)
            if nexts:
                route.append(random.choice(nexts))
            else:
                # Dead end: backtrack để quay đầu
                # Curve factor sẽ tự giảm tốc vì góc backtrack ≈ 180°
                back = route[-1]
                for _ in range(100):
                    prevs = back.previous(2.0)
                    if not prevs:
                        break
                    back = random.choice(prevs)
                    route.append(back)
                break

        self._routes[vehicle.id]        = [w.transform.location for w in route]
        self._offroad_ticks[vehicle.id] = 0

    def update(self, vehicle):
        """Gọi mỗi tick: navigate + speed control + off-road detection."""
        if vehicle.id not in self._routes:
            self.register(vehicle)
            self.assign_route(vehicle)

        loc = vehicle.get_location()

        # ── Off-road detection (distance-based) ─────────────
        # project_to_road=True luôn snap về đường gần nhất.
        # Nếu khoảng cách snap > 4m → xe đã ra ngoài road boundary.
        snap_wp = self._map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
        off_road = (snap_wp is None or
                    _dist2d(loc, snap_wp.transform.location) > 4.0)

        if off_road:
            ticks = self._offroad_ticks.get(vehicle.id, 0) + 1
            self._offroad_ticks[vehicle.id] = ticks
            vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0,
                                                        hand_brake=True))
            if ticks >= 3:
                # Teleport về spawn gần nhất trên đường —
                # chỉ brake không đủ vì xe cố navigate từ vị trí off-road
                spawn_points = self._map.get_spawn_points()
                if spawn_points:
                    nearest = min(spawn_points,
                                  key=lambda sp: _dist2d(loc, sp.location))
                    vehicle.set_transform(nearest)
                self._offroad_ticks[vehicle.id] = 0
                self.assign_route(vehicle)
            return
        self._offroad_ticks[vehicle.id] = 0
        # ────────────────────────────────────────────────────

        route = self._routes[vehicle.id]

        # Pop waypoint đã qua
        while route and _dist2d(loc, route[0]) < self.LOOKAHEAD_DIST * 0.5:
            route.pop(0)

        # Hết route → tái tạo
        if not route:
            self.assign_route(vehicle)
            route = self._routes.get(vehicle.id, [])
            if not route:
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
                return

        # Pure-pursuit steering
        target = route[0]
        fwd    = vehicle.get_transform().get_forward_vector()
        dx     = target.x - loc.x
        dy     = target.y - loc.y
        dist   = max(1e-3, math.sqrt(dx * dx + dy * dy))
        cross  = fwd.x * (dy / dist) - fwd.y * (dx / dist)
        steer  = max(-1.0, min(1.0, cross * 2.0))

        # Tốc độ mục tiêu — curve factor tự xử lý U-turn (góc ~180° → factor 0.25)
        target_spd = self._compute_target_speed(vehicle.id, route)

        # Speed control
        vel   = vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        error = target_spd - speed

        if error > 0:
            throttle = min(0.75, error * 0.35)
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = min(1.0, -error * 0.5)

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        ))

    def reset_actor(self, actor_id: int):
        for d in (self._routes, self._speed_limit, self._personality,
                  self._target_speed, self._speed_jitter, self._offroad_ticks):
            d.pop(actor_id, None)

    # ----------------------------------------------------------
    # Speed computation
    # ----------------------------------------------------------

    def _compute_target_speed(self, actor_id: int, route: list) -> float:
        """
        Tính tốc độ mục tiêu với 3 thành phần:
        1. Base speed = speed_limit × personality
        2. Curve factor: giảm tốc khi phát hiện cua/U-turn phía trước
           (góc 180° khi quay đầu → factor 0.25 → ~15 km/h tự động)
        3. Jitter: dao động nhỏ ±8% mỗi 2–5 giây — mô phỏng nhấp ga thực tế
        """
        limit  = self._speed_limit.get(actor_id, self._DEFAULT_SPEED)
        pers   = self._personality.get(actor_id, 0.75)
        base   = limit * pers

        # Curve factor
        max_angle  = self._lookahead_max_angle(route)
        # Cua nhẹ (<15°): không giảm; cua 45°: ~70%; cua 90°: ~40%; quay đầu: ~25%
        curve_factor = max(0.25, 1.0 - max_angle / 130.0)
        raw_target   = base * curve_factor

        # Speed jitter: mỗi xe dao động ngẫu nhiên để tránh tất cả xe
        # cùng tốc độ đều nhau
        timer, jitter = self._speed_jitter.get(actor_id, (0, 0.0))
        timer += 1
        if timer >= random.randint(20, 50):   # 2–5 giây @ 10Hz
            jitter = random.uniform(-0.08, 0.08) * base
            timer  = 0
        self._speed_jitter[actor_id] = (timer, jitter)

        raw_target = max(0.5, raw_target + jitter)

        # Smooth: 85% prev + 15% new → gia tốc/phanh không tức thì
        prev  = self._target_speed.get(actor_id, base)
        final = 0.85 * prev + 0.15 * raw_target
        self._target_speed[actor_id] = final
        return final

    def _lookahead_max_angle(self, route: list) -> float:
        """Góc rẽ lớn nhất (degrees) trong CURVE_LOOKAHEAD waypoints phía trước."""
        n = min(self.CURVE_LOOKAHEAD, len(route) - 1)
        if n < 2:
            return 0.0

        max_angle = 0.0
        for i in range(n - 1):
            dx1 = route[i + 1].x - route[i].x
            dy1 = route[i + 1].y - route[i].y
            dx2 = route[i + 2].x - route[i + 1].x if i + 2 < len(route) else dx1
            dy2 = route[i + 2].y - route[i + 1].y if i + 2 < len(route) else dy1

            len1 = max(1e-6, math.sqrt(dx1 ** 2 + dy1 ** 2))
            len2 = max(1e-6, math.sqrt(dx2 ** 2 + dy2 ** 2))
            cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
            max_angle = max(max_angle, angle)

        return max_angle


# ──────────────────────────────────────────────────────────
# Traffic Generator
# ──────────────────────────────────────────────────────────

class TrafficGenerator:
    """
    Generates traffic actors (vehicles and pedestrians) for testing.
    Vehicles dùng WaypointFollower với hành vi lái thực tế.
    """

    # Tỉ lệ phương tiện mặc định — phù hợp traffic Hà Nội (Mỗ Lao khu vực)
    # Thực tế: ~70-75% xe máy, ~20% ô tô, phần còn lại xe lớn
    DEFAULT_VEHICLE_MIX = {
        "motorcycle": 0.70,
        "car":        0.20,
        "truck":      0.05,
        "bus":        0.05,
    }

    def __init__(self, world, num_vehicles=10, num_pedestrians=5, vehicle_mix=None):
        self.world           = world
        self.num_vehicles    = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.vehicle_list    = []
        self.pedestrian_list = []
        self._follower       = WaypointFollower(world.get_map())
        # vehicle_mix: dict[class_name → ratio], mặc định theo Hà Nội
        self.vehicle_mix     = vehicle_mix or self.DEFAULT_VEHICLE_MIX

        logger.info(f"TrafficGenerator initialized: {num_vehicles} vehicles, "
                    f"{num_pedestrians} pedestrians, mix={self.vehicle_mix}")

    def spawn_actors(self):
        """Spawn vehicles và pedestrians tại random spawn points."""
        blueprint_library = self.world.get_blueprint_library()
        spawn_points      = self.world.get_map().get_spawn_points()

        if not spawn_points:
            logger.warning("No spawn points available on map")
            return

        self._spawn_vehicles(blueprint_library, spawn_points)
        self._spawn_pedestrians(blueprint_library, spawn_points)

        # Tick để CARLA đăng ký actors
        self.world.tick()

        # Đăng ký và gán route ngay sau khi spawn
        for v in self.vehicle_list:
            self._follower.register(v)
            self._follower.assign_route(v)

        logger.info(f"Spawned {len(self.vehicle_list)} vehicles and {len(self.pedestrian_list)} pedestrians")

    def _build_blueprint_pools(self, blueprint_library):
        """
        Phân loại vehicle blueprints theo class, trả về dict[class → list[bp]].
        Dùng _classify_type_id() giống collect script để đảm bảo nhất quán.
        """
        pools = {cls: [] for cls in self.vehicle_mix}
        for bp in blueprint_library.filter('vehicle.*'):
            cls = _classify_type_id(bp.id)
            if cls in pools:
                pools[cls].append(bp)
        # Log để debug
        for cls, bps in pools.items():
            logger.info(f"  Blueprint pool [{cls}]: {len(bps)} blueprints")
        return pools

    def _pick_blueprint(self, pools):
        """
        Chọn blueprint theo vehicle_mix (weighted random).
        Nếu pool của class nào đó rỗng → redistribute weight sang class còn lại.
        """
        available = {cls: bps for cls, bps in pools.items() if bps}
        if not available:
            return None
        # Tính weight chỉ cho các class có blueprint
        total = sum(self.vehicle_mix.get(cls, 0) for cls in available)
        if total == 0:
            cls = random.choice(list(available.keys()))
        else:
            r = random.random() * total
            acc = 0.0
            cls = list(available.keys())[-1]
            for c in available:
                acc += self.vehicle_mix.get(c, 0)
                if r <= acc:
                    cls = c
                    break
        return random.choice(available[cls])

    def _spawn_vehicles(self, blueprint_library, spawn_points):
        """Spawn vehicles theo vehicle_mix — không dùng autopilot."""
        pools = self._build_blueprint_pools(blueprint_library)
        for _ in range(self.num_vehicles):
            try:
                bp = self._pick_blueprint(pools)
                if bp is None:
                    logger.warning("No blueprint available, skipping")
                    continue
                if bp.has_attribute('color'):
                    bp.set_attribute('color',
                                     random.choice(bp.get_attribute('color').recommended_values))
                vehicle = self.world.try_spawn_actor(bp, random.choice(spawn_points))
                if vehicle is not None:
                    # Brake ngay để tránh CARLA physics spawn glitch (velocity spike)
                    vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                    self.vehicle_list.append(vehicle)
                else:
                    logger.warning("Failed to spawn vehicle")
            except Exception as e:
                logger.error(f"Error spawning vehicle: {e}")

    def _spawn_pedestrians(self, blueprint_library, spawn_points):
        """Spawn pedestrians với WalkerControl trực tiếp.

        Không dùng controller.ai.walker: RPC 'set_actor_collisions'
        không tồn tại ở server CARLA 0.9.14 → world.tick() lỗi vĩnh viễn.
        """
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        for _ in range(self.num_pedestrians):
            try:
                bp         = random.choice(walker_blueprints)
                pedestrian = self.world.try_spawn_actor(bp, random.choice(spawn_points))
                if pedestrian is not None:
                    self.pedestrian_list.append(pedestrian)
                    self._walk_to_nearby_target(pedestrian, spawn_points)
                else:
                    logger.warning("Failed to spawn pedestrian")
            except Exception as e:
                logger.error(f"Error spawning pedestrian: {e}")

    def _walk_to_nearby_target(self, pedestrian, spawn_points):
        """Cho pedestrian đi về spawn point gần (<300m) — tránh đi thẳng ra ngoài map."""
        try:
            loc    = pedestrian.get_location()
            nearby = [sp for sp in spawn_points if _dist2d(loc, sp.location) < 300.0]
            target = random.choice(nearby if nearby else spawn_points).location
            dx     = target.x - loc.x
            dy     = target.y - loc.y
            dist   = max(1e-3, math.sqrt(dx ** 2 + dy ** 2))
            speed  = random.uniform(1.0, 2.2)   # 3.6–7.9 km/h — đi bộ tự nhiên
            pedestrian.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0),
                speed=speed,
            ))
        except Exception as e:
            logger.error(f"Error setting pedestrian walk control: {e}")

    def update_vehicles(self):
        """Gọi mỗi tick: WaypointFollower update cho tất cả vehicles."""
        dead = []
        for v in self.vehicle_list:
            try:
                if not v.is_alive:
                    dead.append(v)
                    self._follower.reset_actor(v.id)
                    continue
                self._follower.update(v)
            except Exception:
                dead.append(v)
                self._follower.reset_actor(v.id)
        for v in dead:
            self.vehicle_list.remove(v)

    def update_pedestrians(self):
        """Retarget pedestrians mỗi ~10s (100 frames @ 10Hz)."""
        self._pedestrian_tick = getattr(self, '_pedestrian_tick', 0) + 1
        if self._pedestrian_tick % 100 != 0:
            return
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return
        for pedestrian in self.pedestrian_list:
            self._walk_to_nearby_target(pedestrian, spawn_points)

    def get_actor_info(self):
        return {'vehicles': len(self.vehicle_list), 'pedestrians': len(self.pedestrian_list)}

    def cleanup(self):
        """Destroy all spawned actors."""
        logger.info("Cleaning up traffic actors...")
        for vehicle in self.vehicle_list:
            try:
                vehicle.destroy()
            except Exception as e:
                logger.error(f"Error destroying vehicle: {e}")
        for pedestrian in self.pedestrian_list:
            try:
                pedestrian.destroy()
            except Exception as e:
                logger.error(f"Error destroying pedestrian: {e}")
        self.vehicle_list.clear()
        self.pedestrian_list.clear()
        logger.info("Traffic cleanup completed")
