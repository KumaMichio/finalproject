"""
Test ve buildings dang "khoi xam" (buildings_render.draw_buildings): outline
footprint + noc + canh doc + hatch-fill xam, chup anh top-down de kiem tra.

Chay: venv_tracking\\Scripts\\python.exe Map\\test_buildings_grey.py
Yeu cau: CARLA dang chay, world hien tai la map3 (hanoi_district3.xodr)
"""
import json
import time

import carla

from buildings_render import draw_buildings

HOST = "127.0.0.1"
PORT = 2000
BUILDINGS_FILE = "buildings_hanoi_district3.json"
OUT_IMG = "map3_buildings_grey_topdown.png"


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()

    with open(BUILDINGS_FILE, "r", encoding="utf-8") as f:
        buildings = json.load(f)
    print(f"Loaded {len(buildings)} buildings")

    draw_buildings(world, buildings, life_time=30.0)
    time.sleep(0.5)

    wps = carla_map.generate_waypoints(distance=10.0)
    xs = [b["center"][0] for b in buildings] + [wp.transform.location.x for wp in wps]
    ys = [b["center"][1] for b in buildings] + [wp.transform.location.y for wp in wps]
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
