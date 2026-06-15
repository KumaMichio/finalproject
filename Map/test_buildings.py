"""
Ve building footprints (buildings_hanoi_district3.json) bang debug box
trong CARLA, dat spectator top-down, chup anh de doi chieu voi road network.

Chay: venv_tracking\\Scripts\\python.exe Map\\test_buildings.py
Yeu cau: CARLA dang chay, world hien tai la map3 (hanoi_district3.xodr)
"""
import json
import math
import time

import carla

HOST = "127.0.0.1"
PORT = 2000
BUILDINGS_FILE = "buildings_hanoi_district3.json"
OUT_IMG = "map3_buildings_topdown.png"


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    print(f"Map: {world.get_map().name}")

    with open(BUILDINGS_FILE, "r", encoding="utf-8") as f:
        buildings = json.load(f)
    print(f"Loaded {len(buildings)} buildings")

    debug = world.debug

    life_time = 30.0  # giay - du de chup anh
    for b in buildings:
        cx, cy = b["center"]
        w, h = b["size"]
        height = b["height"]
        yaw = b["yaw_deg"]

        location = carla.Location(x=cx, y=cy, z=height / 2.0)
        rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)
        extent = carla.Vector3D(x=w / 2.0, y=h / 2.0, z=height / 2.0)
        box = carla.BoundingBox(location, extent)

        debug.draw_box(
            box, rotation,
            thickness=0.3,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )

    # Tinh bounding box tong de dat spectator top-down
    xs = [b["center"][0] for b in buildings]
    ys = [b["center"][1] for b in buildings]
    # Bao gom ca road waypoints de canh chinh voi map3_topdown.png
    carla_map = world.get_map()
    wps = carla_map.generate_waypoints(distance=10.0)
    xs += [wp.transform.location.x for wp in wps]
    ys += [wp.transform.location.y for wp in wps]

    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys))

    spectator = world.get_spectator()
    height_cam = span * 0.6 + 50
    transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=height_cam),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    spectator.set_transform(transform)
    print(f"Spectator: center=({cx:.1f},{cy:.1f}) height={height_cam:.1f}")

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "1280")
    cam_bp.set_attribute("fov", "90")

    camera = world.spawn_actor(cam_bp, transform)

    captured = {"done": False}

    def on_image(image):
        if not captured["done"]:
            image.save_to_disk(OUT_IMG)
            captured["done"] = True

    camera.listen(on_image)

    timeout = time.time() + 15
    while not captured["done"] and time.time() < timeout:
        time.sleep(0.2)

    camera.stop()
    camera.destroy()

    if captured["done"]:
        print(f"Da luu anh: {OUT_IMG}")
    else:
        print("Khong chup duoc anh")


if __name__ == "__main__":
    main()
