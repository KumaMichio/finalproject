#!/usr/bin/env python3
"""
CARLA Detection Dataset Collector v2 — 6 intersection cameras

Cải thiện so với v1 (collect_carla_cam_data.py):
  1. TrafficManager autopilot thay WaypointFollower (v1 có bug: update_vehicles()
     không được gọi trong loop → xe đứng im → ảnh toàn đường trống)
  2. 60 vehicles mặc định, 70% xe máy (Hà Nội traffic mix)
  3. Lọc ảnh trống: chỉ lưu frame khi có ≥ MIN_BOXES box trong camera
  4. Warmup 200 ticks trước khi capture (để xe phân tán khắp map)
  5. 6 cameras, cao z=9, đặt ở góc chéo ngã tư (yaw=45/225°), pitch=-25°
  6. Log thống kê per-camera mỗi 100 ticks để kiểm tra coverage
  7. Capture adaptive: lưu nhiều hơn khi có xe, bỏ qua ticks ít xe

Usage:
    python collect_carla_cam_det_v2.py --frames 5000 --num-vehicles 60 --out data/carla_cam_det_v2

    # Nếu muốn chạy nhanh hơn (ít frame hơn nhưng đủ data):
    python collect_carla_cam_det_v2.py --frames 3000 --num-vehicles 80 --out data/carla_cam_det_v2

Output:
    data/carla_cam_det_v2/images/{train,val}/*.jpg
    data/carla_cam_det_v2/labels/{train,val}/*.txt
    data/carla_cam_det_v2.yaml
"""

import argparse
import logging
import queue
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml as pyyaml

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.calibration import CameraCalibration

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_FIXED_DELTA = 0.1   # 10 Hz

VN_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']
VN_PERSON, VN_CAR, VN_MOTORCYCLE, VN_BUS, VN_TRUCK = range(5)

# Blueprint classification
_BICYCLE_HINTS = frozenset(['crossbike', 'omafiets', 'diamondback', 'gazelle', 'bh.'])
_MOTO_HINTS    = frozenset(['harley-davidson', 'kawasaki', 'yamaha', 'vespa',
                             'ninja', 'low_rider', 'zx125', 'ysf'])
_BUS_HINTS     = frozenset(['fusorosa', 'sprinter', 'futura', 'volkswagen.t2'])
_TRUCK_HINTS   = frozenset(['carlacola', 'cybertruck', 'firetruck', 'ambulance'])

# Hanoi traffic mix
VEHICLE_MIX = {'motorcycle': 0.70, 'car': 0.25, 'truck': 0.03, 'bus': 0.02}

MIN_BOX_PX   = 4     # pixel kích thước tối thiểu để coi là valid box
MIN_BOXES    = 3     # bỏ qua frame nếu số box < MIN_BOXES


# ─────────────────────────── CARLA helpers ──────────────────────────────────

def connect_carla(host, port):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = _FIXED_DELTA
    world.apply_settings(settings)

    logger.info("Connected: map=%s  synchronous @%.0fHz",
                world.get_map().name, 1.0 / _FIXED_DELTA)
    return client, world


def classify_bp(bp) -> str:
    tid = bp.id.lower()
    if any(h in tid for h in _BICYCLE_HINTS):
        return 'bicycle'
    if any(h in tid for h in _MOTO_HINTS):
        return 'motorcycle'
    if any(h in tid for h in _BUS_HINTS):
        return 'bus'
    if any(h in tid for h in _TRUCK_HINTS):
        return 'truck'
    return 'car'


def classify_vehicle(actor) -> int:
    type_id = actor.type_id.lower()
    if any(h in type_id for h in _MOTO_HINTS):
        return VN_MOTORCYCLE
    if any(h in type_id for h in _BUS_HINTS):
        return VN_BUS
    if any(h in type_id for h in _TRUCK_HINTS):
        return VN_TRUCK
    return VN_CAR


def build_bp_pools(world):
    pools = {k: [] for k in VEHICLE_MIX}
    for bp in world.get_blueprint_library().filter('vehicle.*'):
        cls = classify_bp(bp)
        if cls == 'bicycle':
            continue
        if cls in pools:
            pools[cls].append(bp)
    for cls, bps in pools.items():
        logger.info("  Blueprint pool [%s]: %d entries", cls, len(bps))
    return pools


def pick_bp(pools: dict, rng: random.Random):
    available = {c: bps for c, bps in pools.items() if bps}
    if not available:
        return None
    total  = sum(VEHICLE_MIX.get(c, 0) for c in available)
    if total == 0:
        cls = rng.choice(list(available))
    else:
        r, acc = rng.random() * total, 0.0
        cls = list(available)[-1]
        for c in available:
            acc += VEHICLE_MIX.get(c, 0)
            if r <= acc:
                cls = c
                break
    return rng.choice(available[cls])


_INTERSECTION_CENTERS = [
    (-119.2, 12.0),   # A — western  (North TL at -119.2, 5.1)
    ( -63.3, 13.6),   # B — middle
    ( -31.8, 26.9),   # C — eastern
]
_INTERSECTION_RADIUS_M = 80   # spawn points within this radius are "near-intersection"


def _near_any_intersection(sp, radius: float = _INTERSECTION_RADIUS_M) -> bool:
    import math
    x, y = sp.location.x, sp.location.y
    return any(math.sqrt((x - ix)**2 + (y - iy)**2) <= radius
               for ix, iy in _INTERSECTION_CENTERS)


def spawn_traffic(client, world, num_vehicles: int, tm_port: int, rng: random.Random):
    """Spawn vehicles near target intersections using TM autopilot.

    Prioritises spawn points within 80m of any of the 3 monitored intersections so
    vehicles are always visible in camera frames. Falls back to map-wide spawn points
    if not enough near-intersection points are available.

    NOTE: client.get_trafficmanager() crashes with CARLA 0.9.14 server + 0.9.15 client
    (STATUS_STACK_BUFFER_OVERRUN). Use set_autopilot(True) without explicit TM port.
    """
    import carla

    pools = build_bp_pools(world)
    carla_map = world.get_map()
    all_spawn_pts = carla_map.get_spawn_points()
    if not all_spawn_pts:
        logger.error("No spawn points available — is Town10HD_Opt loaded?")
        return []

    # Partition: near-intersection (priority) vs rest-of-map (fallback)
    near_pts  = [sp for sp in all_spawn_pts if _near_any_intersection(sp)]
    other_pts = [sp for sp in all_spawn_pts if not _near_any_intersection(sp)]
    logger.info("Spawn points: %d near intersections, %d elsewhere (total %d)",
                len(near_pts), len(other_pts), len(all_spawn_pts))

    # Shuffle both pools, use near-intersection first
    rng.shuffle(near_pts)
    rng.shuffle(other_pts)
    candidate_pts = near_pts + other_pts

    vehicles = []
    for sp in candidate_pts[:num_vehicles * 3]:
        if len(vehicles) >= num_vehicles:
            break
        bp = pick_bp(pools, rng)
        if bp is None:
            continue
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(
                bp.get_attribute('color').recommended_values))
        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue
        actor.set_autopilot(True)
        vehicles.append(actor)

    logger.info("Spawned %d/%d vehicles (near-intersection spawn points: %d/%d used)",
                len(vehicles), num_vehicles,
                min(len(vehicles), len(near_pts)), len(near_pts))
    return vehicles


# ──────────────────────────── Camera ─────────────────────────────────────────

class IntersectionCamera:
    """RGB camera synced qua per-camera queue."""

    def __init__(self, world, cfg: dict):
        import carla
        self.camera_id = cfg['camera_id']
        self.w, self.h = cfg['resolution']
        self.calib = CameraCalibration.from_carla_params(
            position=cfg['position'],
            rotation=cfg['rotation'],
            fov=cfg['view_angle'],
            resolution=cfg['resolution'],
        )

        bp_lib = world.get_blueprint_library()
        transform = carla.Transform(
            carla.Location(*cfg['position']),
            carla.Rotation(pitch=cfg['rotation'][0],
                           yaw=cfg['rotation'][1],
                           roll=cfg['rotation'][2]))

        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(self.w))
        rgb_bp.set_attribute('image_size_y', str(self.h))
        rgb_bp.set_attribute('fov',          str(cfg['view_angle']))
        self._sensor = world.spawn_actor(rgb_bp, transform)
        self._q = queue.Queue()
        self._sensor.listen(self._q.put)

        logger.info("Camera %s: pos=%s rot=%s fov=%s",
                    self.camera_id, cfg['position'],
                    cfg['rotation'], cfg['view_angle'])

    def get_frame(self, frame_id: int, timeout=2.0):
        try:
            while True:
                img = self._q.get(timeout=timeout)
                if img.frame == frame_id:
                    arr = np.frombuffer(img.raw_data, dtype=np.uint8)
                    return arr.reshape((self.h, self.w, 4))[:, :, :3]
                if img.frame > frame_id:
                    return None
        except queue.Empty:
            return None

    def destroy(self):
        try:
            self._sensor.stop()
            self._sensor.destroy()
        except Exception:
            pass


# ───────────────────────── Bbox projection ───────────────────────────────────

def _world_corners(actor):
    import carla
    bb, tf = actor.bounding_box, actor.get_transform()
    ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                loc = carla.Location(
                    bb.location.x + sx * ex,
                    bb.location.y + sy * ey,
                    bb.location.z + sz * ez)
                w = tf.transform(loc)
                pts.append((w.x, w.y, w.z))
    return pts


def project_bbox(cam: IntersectionCamera, actor):
    px = []
    for wx, wy, wz in _world_corners(actor):
        p = cam.calib.project_point((wx, wy, wz))
        if p is not None:
            px.append(p)
    if len(px) < 2:
        return None
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    x1 = max(0.0,       min(xs))
    x2 = min(float(cam.w), max(xs))
    y1 = max(0.0,       min(ys))
    y2 = min(float(cam.h), max(ys))
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    return x1, y1, x2, y2


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


# ─────────────────────────── Main ────────────────────────────────────────────

def load_cam_configs(config_path: str) -> list:
    with open(config_path, encoding='utf-8') as f:
        cfg = pyyaml.safe_load(f)
    return [
        dict(
            camera_id  = v['camera_id'],
            position   = v['position'],
            rotation   = v['rotation'],
            view_angle = v['view_angle'],
            resolution = v['resolution'],
        )
        for v in cfg['cameras'].values()
    ]


def write_yaml(out_root: Path, yaml_path: Path):
    content = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        f"nc: {len(VN_CLASS_NAMES)}\n"
        f"names: {VN_CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host',         default='localhost')
    ap.add_argument('--port',         type=int, default=2000)
    ap.add_argument('--tm-port',      type=int, default=8000)
    ap.add_argument('--frames',       type=int, default=5000,
                    help='Tổng số ticks simulation')
    ap.add_argument('--warmup',       type=int, default=200,
                    help='Ticks warmup trước khi bắt đầu capture (để xe phân tán)')
    ap.add_argument('--capture-every',type=int, default=3,
                    help='Capture 1 ảnh/camera mỗi N ticks')
    ap.add_argument('--min-boxes',    type=int, default=3,
                    help='Bỏ qua frame nếu số box < min-boxes')
    ap.add_argument('--num-vehicles', type=int, default=60,
                    help='Số xe spawn (70%% motorcycle, 25%% car, ...)')
    ap.add_argument('--val-ratio',    type=float, default=0.1)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--config',       default='config/camera_config_6cam.yaml')
    ap.add_argument('--out',          default='data/carla_cam_det_v2')
    ap.add_argument('--yaml',         default='data/carla_cam_det_v2.yaml')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_root = Path(args.out)
    for split in ('train', 'val'):
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    client, world = connect_carla(args.host, args.port)
    vehicles = spawn_traffic(client, world, args.num_vehicles,
                             args.tm_port, rng)

    cam_cfgs = load_cam_configs(args.config)
    logger.info("Loading %d cameras from %s", len(cam_cfgs), args.config)

    cameras = []
    try:
        for cfg in cam_cfgs:
            cameras.append(IntersectionCamera(world, cfg))

        # ── Warmup ────────────────────────────────────────────
        logger.info("Warmup %d ticks — letting vehicles spread across map ...",
                    args.warmup)
        for _ in range(args.warmup):
            world.tick()

        # ── Capture loop ──────────────────────────────────────
        n_images = n_boxes = 0
        cam_box_hist = defaultdict(int)   # per-camera total boxes (for stats)
        cam_img_hist = defaultdict(int)

        t0 = time.time()
        for tick in range(args.frames):
            try:
                world.tick()
            except RuntimeError as e:
                logger.error("world.tick() failed at tick %d: %s", tick, e)
                break

            if tick % args.capture_every != 0:
                continue

            snapshot = world.get_snapshot()
            frame_id = snapshot.frame

            actors   = world.get_actors()
            vehicles_live = list(actors.filter('vehicle.*'))
            walkers  = list(actors.filter('walker.pedestrian.*'))
            split    = 'val' if rng.random() < args.val_ratio else 'train'

            for cam in cameras:
                rgb = cam.get_frame(frame_id)
                if rgb is None:
                    continue

                lines = []
                for actor in vehicles_live:
                    box = project_bbox(cam, actor)
                    if box is None:
                        continue
                    cls = classify_vehicle(actor)
                    cx, cy, w, h = to_yolo(box, cam.w, cam.h)
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                for actor in walkers:
                    box = project_bbox(cam, actor)
                    if box is None:
                        continue
                    cx, cy, w, h = to_yolo(box, cam.w, cam.h)
                    lines.append(f"{VN_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # Bỏ qua frame ít xe — đây là fix chính cho "ảnh đường trống"
                if len(lines) < args.min_boxes:
                    continue

                stem     = f"carla_{frame_id:08d}_{cam.camera_id}"
                img_path = out_root / 'images' / split / f"{stem}.jpg"
                lbl_path = out_root / 'labels' / split / f"{stem}.txt"
                cv2.imwrite(str(img_path), rgb)
                lbl_path.write_text('\n'.join(lines) + '\n')

                n_images += 1
                n_boxes  += len(lines)
                cam_box_hist[cam.camera_id] += len(lines)
                cam_img_hist[cam.camera_id] += 1

            # Progress log mỗi 100 ticks
            if tick > 0 and tick % 100 == 0:
                elapsed = time.time() - t0
                fps_sim  = tick / elapsed
                logger.info(
                    "tick %d/%d | %d img, %d boxes | %.1f sim-ticks/s",
                    tick, args.frames, n_images, n_boxes, fps_sim)
                for cam in cameras:
                    cid = cam.camera_id
                    logger.info("  %s: %d images, %d boxes (avg %.1f box/img)",
                                cid, cam_img_hist[cid], cam_box_hist[cid],
                                cam_box_hist[cid] / max(1, cam_img_hist[cid]))

    finally:
        logger.info("Cleaning up cameras and vehicles ...")
        for cam in cameras:
            cam.destroy()
        for v in vehicles:
            try:
                v.destroy()
            except Exception:
                pass

    write_yaml(out_root, Path(args.yaml))

    logger.info("=" * 60)
    logger.info("DONE: %d images, %d boxes  →  %s", n_images, n_boxes, out_root)
    logger.info("Per-camera summary:")
    for cam in cameras:
        cid = cam.camera_id
        n_i, n_b = cam_img_hist[cid], cam_box_hist[cid]
        logger.info("  %s: %d images, %d boxes, avg %.1f box/img",
                    cid, n_i, n_b, n_b / max(1, n_i))
    logger.info("Dataset config: %s", args.yaml)


if __name__ == '__main__':
    main()
