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

    def __init__(self, world, client):
        import carla
        self._carla = carla
        self.world  = world
        self.client = client
        self.tm     = client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)

        self._sensors:  list = []   # collision sensors
        self._actors:   list = []   # actors do scenario tạo ra
        self.collision_log: list = []

        logger.info("ScenarioController initialized")

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
            return None, None

        # Chọn 2 điểm gần nhau, cùng hướng
        pt_b = random.choice(spawn_points)
        pt_a = self._find_nearby_spawn(spawn_points, pt_b, distance=20)

        victim   = self._spawn_vehicle(pt_b, autopilot=True)
        attacker = self._spawn_vehicle(pt_a, autopilot=False)

        if not victim or not attacker:
            return None, None

        self._attach_collision_sensor(attacker, tag='hit_and_run')

        logger.info(f"[HIT_AND_RUN] Victim={victim.id}, Attacker={attacker.id} — "
                    f"va chạm sau {delay_s}s")

        def _run():
            time.sleep(delay_s)
            # Xe A: đâm thẳng về phía trước
            attacker.set_autopilot(True)
            self.tm.vehicle_percentage_speed_difference(attacker, -80)  # nhanh hơn 80%
            self.tm.distance_to_leading_vehicle(attacker, 0)
            self.tm.ignore_vehicles_percentage(attacker, 100)
            logger.info("[HIT_AND_RUN] Attacker đang tiến tới nạn nhân...")

        threading.Thread(target=_run, daemon=True).start()
        return attacker, victim

    def pedestrian_hit(self, delay_s: float = 2.0):
        """Xe tốc độ cao áp sát và đâm người đi bộ."""
        spawn_points = self.world.get_map().get_spawn_points()
        bp_library   = self.world.get_blueprint_library()

        if not spawn_points:
            return None, None

        pt = random.choice(spawn_points)
        vehicle    = self._spawn_vehicle(pt, autopilot=True)
        pedestrian = self._spawn_pedestrian(bp_library)

        if not vehicle:
            return None, None

        self._attach_collision_sensor(vehicle, tag='pedestrian_hit')

        def _run():
            time.sleep(delay_s)
            self.tm.ignore_walkers_percentage(vehicle, 100)
            self.tm.vehicle_percentage_speed_difference(vehicle, -40)
            logger.info("[PEDESTRIAN_HIT] Vehicle đang nhắm vào người đi bộ...")

        threading.Thread(target=_run, daemon=True).start()
        return vehicle, pedestrian

    def red_light_crash(self, delay_s: float = 2.0):
        """Hai xe vượt đèn đỏ từ hai hướng đối diện."""
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            return None, None

        pt_a = random.choice(spawn_points)
        pt_b = self._find_opposite_spawn(spawn_points, pt_a)

        v_a = self._spawn_vehicle(pt_a, autopilot=True)
        v_b = self._spawn_vehicle(pt_b, autopilot=True)

        if not v_a or not v_b:
            return None, None

        self._attach_collision_sensor(v_a, tag='red_light_crash')
        self._attach_collision_sensor(v_b, tag='red_light_crash')

        def _run():
            time.sleep(delay_s)
            for v in (v_a, v_b):
                self.tm.ignore_lights_percentage(v, 100)
                self.tm.vehicle_percentage_speed_difference(v, -30)
                self.tm.ignore_vehicles_percentage(v, 100)
            logger.info("[RED_LIGHT_CRASH] Hai xe đang vượt đèn đỏ...")

        threading.Thread(target=_run, daemon=True).start()
        return v_a, v_b

    def rear_end(self, delay_s: float = 2.0):
        """Xe đâm từ phía sau xe đang chạy chậm."""
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            return None, None

        pt_front = random.choice(spawn_points)
        pt_rear  = self._find_nearby_spawn(spawn_points, pt_front, distance=15)

        front_car = self._spawn_vehicle(pt_front, autopilot=True)
        rear_car  = self._spawn_vehicle(pt_rear,  autopilot=True)

        if not front_car or not rear_car:
            return None, None

        self._attach_collision_sensor(rear_car, tag='rear_end')

        def _run():
            time.sleep(delay_s)
            # Xe trước chạy chậm
            self.tm.vehicle_percentage_speed_difference(front_car, 50)
            # Xe sau tăng tốc, không giữ khoảng cách
            self.tm.vehicle_percentage_speed_difference(rear_car, -60)
            self.tm.distance_to_leading_vehicle(rear_car, 0)
            logger.info("[REAR_END] Xe phía sau đang đâm vào xe phía trước...")

        threading.Thread(target=_run, daemon=True).start()
        return rear_car, front_car

    def sudden_stop(self, delay_s: float = 3.0):
        """Xe dừng đột ngột giữa đường."""
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return None

        pt = random.choice(spawn_points)
        vehicle = self._spawn_vehicle(pt, autopilot=True)
        if not vehicle:
            return None

        def _run():
            time.sleep(delay_s)
            vehicle.set_autopilot(False)
            vehicle.apply_control(self._carla.VehicleControl(
                throttle=0.0, brake=1.0, steer=0.0
            ))
            logger.info(f"[SUDDEN_STOP] Xe #{vehicle.id} dừng đột ngột")

        threading.Thread(target=_run, daemon=True).start()
        return vehicle

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

    def _spawn_pedestrian(self, bp_library):
        walker_bps = bp_library.filter('walker.pedestrian.*')
        ctrl_bp    = bp_library.find('controller.ai.walker')
        spawn_pts  = self.world.get_map().get_spawn_points()
        if not walker_bps or not spawn_pts:
            return None

        bp  = random.choice(list(walker_bps))
        pt  = random.choice(spawn_pts)
        ped = self.world.try_spawn_actor(bp, pt)
        if ped:
            self.world.tick()  # cần tick trước khi spawn controller
            ctrl = self.world.spawn_actor(ctrl_bp, self._carla.Transform(), attach_to=ped)
            if ctrl:
                ctrl.start()
                ctrl.go_to_location(random.choice(spawn_pts).location)
                ctrl.set_max_speed(1.2)
                self._actors.append(ctrl)
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
