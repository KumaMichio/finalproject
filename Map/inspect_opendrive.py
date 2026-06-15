"""
Kiem tra map OpenDRIVE da load: tinh bounding box cua waypoints,
dat spectator + 1 camera nhin tu tren xuong, chup anh de xem map
thuc te render ra sao.

Chay: venv_tracking\\Scripts\\python.exe Map\\inspect_opendrive.py
Yeu cau: CARLA dang chay va da generate_opendrive_world (script truoc)
"""
import time
import carla
import numpy as np

HOST = "127.0.0.1"
PORT = 2000
OUT_IMG = "map_topdown.png"


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()

    carla_map = world.get_map()
    print(f"Map name: {carla_map.name}")

    waypoints = carla_map.generate_waypoints(distance=10.0)
    xs = np.array([wp.transform.location.x for wp in waypoints])
    ys = np.array([wp.transform.location.y for wp in waypoints])
    zs = np.array([wp.transform.location.z for wp in waypoints])

    print(f"So waypoints: {len(waypoints)}")
    print(f"X range: {xs.min():.1f} .. {xs.max():.1f} (rong {xs.max()-xs.min():.1f}m)")
    print(f"Y range: {ys.min():.1f} .. {ys.max():.1f} (rong {ys.max()-ys.min():.1f}m)")
    print(f"Z range: {zs.min():.1f} .. {zs.max():.1f}")

    cx, cy = xs.mean(), ys.mean()
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    print(f"Center: ({cx:.1f}, {cy:.1f}), span={span:.1f}m")

    # Dat spectator nhin tu tren xuong toan canh
    spectator = world.get_spectator()
    height = span * 0.6 + 50
    transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=height),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    spectator.set_transform(transform)
    print(f"Spectator dat tai ({cx:.1f}, {cy:.1f}, {height:.1f}), pitch=-90")

    # Spawn 1 camera RGB tai vi tri spectator de chup anh
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

    # cho world tick (async mode) de camera chup
    timeout = time.time() + 15
    while not captured["done"] and time.time() < timeout:
        time.sleep(0.2)

    camera.stop()
    camera.destroy()

    if captured["done"]:
        print(f"Da luu anh top-down: {OUT_IMG}")
    else:
        print("Khong chup duoc anh (timeout)")


if __name__ == "__main__":
    main()
