"""
Test logic traffic-light simulation tren hanoi_district3: tim cac giao lo lon
tu topology, gan 1 den/giao lo voi chu ky Red/Yellow/Green lech pha, ve debug
point mau lien tuc trong > 1 chu ky, chup anh top-down de kiem tra.

Chay: venv_tracking\\Scripts\\python.exe Map\\test_traffic_lights.py
Yeu cau: CARLA dang chay, world hien tai la map3 (hanoi_district3.xodr)
"""
import time

import carla

from traffic_light_sim import TrafficLightSim, junction_traffic_lights

HOST = "127.0.0.1"
PORT = 2000
OUT_IMG = "map3_traffic_lights_topdown.png"
DURATION = 25.0  # giay mo phong (> 1 chu ky 20s)
DT = 0.5


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()

    locations = junction_traffic_lights(carla_map)
    print(f"Tim duoc {len(locations)} giao lo lon -> {len(locations)} den mo phong")

    sim = TrafficLightSim(locations)

    # Spectator top-down (giong test_buildings.py)
    xs = [l.x for l in locations]
    ys = [l.y for l in locations]
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

    # Chay mo phong: ve den moi DT giay; chup anh giua chu ky de co ca
    # den Red va Green trong cung 1 frame
    t = 0.0
    snap_t = sim.cycle * 0.5
    listening = False
    while t < DURATION:
        sim.draw(world, t, life_time=DT + 0.1)
        if not listening and t >= snap_t:
            camera.listen(on_image)
            listening = True
        time.sleep(DT)
        t += DT

    timeout = time.time() + 5
    while not captured["done"] and time.time() < timeout:
        time.sleep(0.2)

    camera.stop()
    camera.destroy()

    print(f"Trang thai cac den tai t={snap_t:.1f}s:")
    for loc in locations[:8]:
        state = sim.get_state_near(loc, snap_t, radius=5.0)
        print(f"  ({loc.x:.1f},{loc.y:.1f}) -> {state}")

    if captured["done"]:
        print(f"Da luu anh: {OUT_IMG}")
    else:
        print("Khong chup duoc anh")


if __name__ == "__main__":
    main()
