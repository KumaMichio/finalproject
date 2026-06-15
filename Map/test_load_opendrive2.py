"""
Test load file OpenDRIVE (.xodr) sinh tu OSM vao CARLA bang
generate_opendrive_world().

Chay: venv_tracking\\Scripts\\python.exe Map\\test_load_opendrive.py
Yeu cau: CARLA server (CarlaUE4.exe) dang chay tai localhost:2000
"""
import sys
import time
import carla

XODR_PATH = "hanoi_district2.xodr"
HOST = "127.0.0.1"
PORT = 2000


def main():
    with open(XODR_PATH, "r", encoding="utf-8") as f:
        xodr_content = f.read()

    print(f"Doc {XODR_PATH}: {len(xodr_content)} bytes")

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    # CARLA UE4 co the can 30-60s de boot, retry ket noi
    for attempt in range(12):
        try:
            print(f"Server version: {client.get_server_version()}")
            break
        except RuntimeError as e:
            print(f"  Chua ket noi duoc ({attempt+1}/12): {e}")
            time.sleep(5)
    else:
        print("Khong ket noi duoc CARLA server sau 60s")
        sys.exit(1)

    print(f"Client version: {client.get_client_version()}")

    print("Dang generate world tu OpenDRIVE (co the mat 30-60s)...")
    t0 = time.time()

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
        enable_pedestrian_navigation=False,
    )

    world = client.generate_opendrive_world(
        xodr_content, params, reset_settings=False
    )
    print(f"Generate world xong sau {time.time() - t0:.1f}s")

    carla_map = world.get_map()
    print(f"Map name: {carla_map.name}")

    topology = carla_map.get_topology()
    print(f"So canh topology (road segments): {len(topology)}")

    spawn_points = carla_map.get_spawn_points()
    print(f"So spawn points: {len(spawn_points)}")

    if spawn_points:
        sp = spawn_points[0]
        print(f"Spawn point[0]: location={sp.location}, rotation={sp.rotation}")

    waypoints = carla_map.generate_waypoints(distance=10.0)
    print(f"So waypoints (cach 10m): {len(waypoints)}")


if __name__ == "__main__":
    main()
