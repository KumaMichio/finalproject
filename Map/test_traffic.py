"""
Test spawn traffic (vehicles + autopilot/traffic manager) tren map3
(hanoi_district3.xodr) da load vao CARLA.

Kiem tra:
- Xe co spawn duoc khong (try_spawn_actor co fail nhieu khong)
- Xe co di chuyen theo lane khong (khong bi stuck)
- Co loi/crash khi traffic manager dieu khien tren map tu OpenDRIVE khong

Chay: venv_tracking\\Scripts\\python.exe Map\\test_traffic.py
Yeu cau: CARLA dang chay, world hien tai la map3 (hanoi_district3.xodr)
"""
import sys
import time
import carla

sys.path.insert(0, "../custom_tracking_system")
from modules.traffic_generator import TrafficGenerator

HOST = "127.0.0.1"
PORT = 2000
NUM_VEHICLES = 12
TICKS = 150  # 150 * 0.1s = 15s
DT = 0.1


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = client.get_world()
    print(f"Map: {world.get_map().name}")

    # Enable synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8050)
    tm.set_synchronous_mode(True)

    tg = TrafficGenerator(world, num_vehicles=NUM_VEHICLES, num_pedestrians=0)
    tg.spawn_actors()
    print(f"Spawned: {tg.get_actor_info()}")

    if not tg.vehicle_list:
        print("Khong spawn duoc xe nao!")
        return

    # Track positions over time
    history = {v.id: [] for v in tg.vehicle_list}

    for step in range(TICKS):
        world.tick()
        for v in tg.vehicle_list:
            try:
                loc = v.get_location()
                history[v.id].append((loc.x, loc.y))
            except RuntimeError:
                history[v.id].append(None)

    # Analyze movement
    print("\n--- Ket qua sau 15s ---")
    moved = 0
    stuck = 0
    destroyed = 0
    for v in tg.vehicle_list:
        pts = [p for p in history[v.id] if p is not None]
        if len(pts) < 2:
            destroyed += 1
            print(f"  Xe {v.id} ({v.type_id}): bi destroy/loi som")
            continue
        dist = sum(
            ((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2) ** 0.5
            for i in range(1, len(pts))
        )
        status = "OK (di chuyen)" if dist > 1.0 else "STUCK (khong di chuyen)"
        if dist > 1.0:
            moved += 1
        else:
            stuck += 1
        print(f"  Xe {v.id} ({v.type_id}): di {dist:6.1f}m -> {status}")

    print(f"\nTong: {moved} xe di chuyen, {stuck} xe stuck, {destroyed} xe bi loi")

    # Cleanup
    tg.cleanup()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print("Da cleanup, tat synchronous mode.")


if __name__ == "__main__":
    main()
