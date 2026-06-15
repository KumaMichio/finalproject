"""
Ve buildings (tu buildings_*.json) bang debug.draw_box: 1 box wireframe cho
toan bo khoi nha (footprint + chieu cao) + 1 luoi box nho "to mau" xam tai
mat dat de top-down nhin giong "khoi xam" thay vi chi 1 duong vien mong.

debug.draw_line khong hien thi trong sensor.camera.rgb o build CARLA nay,
chi debug.draw_box moi hien thi duoc (va chi ro tu camera o do cao lon,
~500m, giong cach lay anh top-down trong test_load_opendrive*.py). Render
mesh thuc (static.prop.*, vehicle.*) khong hoat dong o khoang cach gan.
"""
import math

import carla


OUTLINE_COLOR = carla.Color(255, 255, 255)
FILL_COLOR = carla.Color(200, 200, 200)

GRID_SPACING = 3.0   # met giua cac box "to mau" trong luoi
MAX_GRID_PER_SIDE = 6  # tran so box moi chieu / building (gioi han so draw call)


def draw_buildings(world, buildings, life_time=30.0, thickness=0.3):
    """Ve moi building trong `buildings` (list dict tu buildings_*.json).
    life_time=0 -> ve vinh vien (cho den khi world reload)."""
    debug = world.debug
    for b in buildings:
        cx, cy = b["center"]
        w, h = b["size"]
        height = b["height"]
        yaw = b["yaw_deg"]
        rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)

        # Outline toan bo khoi nha (footprint + chieu cao)
        location = carla.Location(x=cx, y=cy, z=height / 2.0)
        extent = carla.Vector3D(x=w / 2.0, y=h / 2.0, z=height / 2.0)
        debug.draw_box(
            carla.BoundingBox(location, extent), rotation,
            thickness=thickness, color=OUTLINE_COLOR, life_time=life_time,
        )

        # Luoi box nho "to mau" xam tai mat dat (footprint)
        n_x = min(MAX_GRID_PER_SIDE, max(1, int(w // GRID_SPACING)))
        n_y = min(MAX_GRID_PER_SIDE, max(1, int(h // GRID_SPACING)))
        rad = math.radians(yaw)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cell_w = w / n_x
        cell_h = h / n_y
        grid_extent = carla.Vector3D(x=cell_w / 2.0 * 0.9, y=cell_h / 2.0 * 0.9, z=0.4)
        for ix in range(n_x):
            for iy in range(n_y):
                lx = -w / 2 + cell_w * (ix + 0.5)
                ly = -h / 2 + cell_h * (iy + 0.5)
                gx = cx + lx * cos_a - ly * sin_a
                gy = cy + lx * sin_a + ly * cos_a
                loc = carla.Location(x=gx, y=gy, z=0.4)
                debug.draw_box(
                    carla.BoundingBox(loc, grid_extent), rotation,
                    thickness=thickness, color=FILL_COLOR, life_time=life_time,
                )
