#!/usr/bin/env python3
"""
CARLA Detection Dataset Collector — production CAM_001/002/003 viewpoints

Same auto-labelling approach as `collect_carla_detection_data.py` (RGB + depth
camera pair, 3D actor bboxes projected to 2D via `modules/calibration.py`,
depth-buffer occlusion check), but the cameras are spawned at the EXACT
position/rotation/fov/resolution of CAM_001/CAM_002/CAM_003 as defined in
`config/camera_config.yaml` — i.e. the same viewpoints `main.py` uses at
runtime. This closes the domain gap between the fine-tuning data and the
live inference frames.

Usage:
    python collect_carla_cam_data.py --frames 3000 --capture-every 5 \
        --num-vehicles 15 --num-pedestrians 8 --out data/carla_cam_det

Output:
    data/carla_cam_det/images/{train,val}/*.jpg
    data/carla_cam_det/labels/{train,val}/*.txt
    data/carla_cam_det.yaml   <- ultralytics dataset config (5 VN classes)
"""

import argparse
import logging
import queue
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml as pyyaml

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.traffic_generator import TrafficGenerator
from modules.calibration import CameraCalibration

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_FIXED_DELTA = 0.1  # seconds -> 10 Hz

# VN classes — must match prepare_visdrone_dataset.py / ObjectDetector.CLASSES_VN
VN_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']
VN_PERSON, VN_CAR, VN_MOTORCYCLE, VN_BUS, VN_TRUCK = range(5)

_MOTORCYCLE_HINTS = (
    'harley-davidson', 'kawasaki', 'yamaha', 'vespa',
    'crossbike', 'omafiets', 'century', 'gazelle', 'diamondback', 'bh',
)
_TRUCK_HINTS = ('carlacola', 'cybertruck', 'firetruck', 'truck')
_BUS_HINTS = ('fusorosa', 'sprinter')

MIN_BOX_PX = 2


def connect_carla(host='localhost', port=2000):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = _FIXED_DELTA
    world.apply_settings(settings)

    logger.info("Connected to CARLA, synchronous mode @ %.0f Hz", 1.0 / _FIXED_DELTA)
    return client, world


def classify_vehicle(actor) -> int:
    type_id = actor.type_id.lower()
    try:
        bp = actor.get_world().get_blueprint_library().find(actor.type_id)
        if bp.has_attribute('base_type'):
            base = bp.get_attribute('base_type').as_str().lower()
            if base in ('motorcycle', 'bicycle'):
                return VN_MOTORCYCLE
            if base == 'truck':
                return VN_TRUCK
            if base == 'bus':
                return VN_BUS
            if base == 'car':
                return VN_CAR
    except Exception:
        pass

    if any(h in type_id for h in _MOTORCYCLE_HINTS):
        return VN_MOTORCYCLE
    if any(h in type_id for h in _TRUCK_HINTS):
        return VN_TRUCK
    if any(h in type_id for h in _BUS_HINTS):
        return VN_BUS
    return VN_CAR


class CctvCamera:
    """One RGB + depth camera pair, synced via per-camera queues."""

    def __init__(self, world, position, rotation, fov, resolution, camera_id):
        import carla
        self.camera_id = camera_id
        self.resolution = resolution
        self.calib = CameraCalibration.from_carla_params(
            position=position, rotation=rotation, fov=fov, resolution=resolution)

        bp_lib = world.get_blueprint_library()
        transform = carla.Transform(
            carla.Location(*position),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]))

        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(resolution[0]))
        rgb_bp.set_attribute('image_size_y', str(resolution[1]))
        rgb_bp.set_attribute('fov', str(fov))
        self.rgb_sensor = world.spawn_actor(rgb_bp, transform)
        self.rgb_queue = queue.Queue()
        self.rgb_sensor.listen(self.rgb_queue.put)

        logger.info("Camera %s: pos=%s rot=%s fov=%.0f res=%s",
                     camera_id, position, rotation, fov, resolution)

    def get_frame(self, frame_id, timeout=2.0):
        rgb_img = self._pop_until(self.rgb_queue, frame_id, timeout)
        if rgb_img is None:
            return None

        w, h = self.resolution
        rgb = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return rgb

    @staticmethod
    def _pop_until(q, frame_id, timeout):
        try:
            while True:
                item = q.get(timeout=timeout)
                if item.frame == frame_id:
                    return item
                if item.frame > frame_id:
                    return None
        except queue.Empty:
            return None

    def destroy(self):
        try:
            self.rgb_sensor.stop()
            self.rgb_sensor.destroy()
        except Exception:
            pass


def actor_world_corners(actor):
    import carla
    bb = actor.bounding_box
    ext = bb.extent
    transform = actor.get_transform()

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = carla.Location(
                    bb.location.x + sx * ext.x,
                    bb.location.y + sy * ext.y,
                    bb.location.z + sz * ext.z)
                world_pt = transform.transform(local)
                corners.append((world_pt.x, world_pt.y, world_pt.z))
    return corners


def project_bbox(camera: CctvCamera, actor):
    """Project an actor's 3D bbox onto the camera image -> (x1,y1,x2,y2) or None.

    Matches `modules.ground_truth.project_actors_to_cameras` (the project's
    own definition of "ground truth" for MOT eval): no occlusion/distance
    filtering, just clip to the image bounds with a 2px minimum size.
    """
    width, height = camera.resolution
    corners = actor_world_corners(actor)

    px = []
    for wx, wy, wz in corners:
        proj = camera.calib.project_point((wx, wy, wz))
        if proj is None:
            continue
        px.append(proj)

    if len(px) < 2:
        return None

    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))

    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None

    return x1, y1, x2, y2


def to_yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def write_yaml(out_root: Path, yaml_path: Path):
    lines = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(VN_CLASS_NAMES)}",
        f"names: {VN_CLASS_NAMES}",
        "",
    ]
    yaml_path.write_text('\n'.join(lines))


def load_cam_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = pyyaml.safe_load(f)
    cams = []
    for cam_name, cfg in config['cameras'].items():
        cams.append(dict(
            camera_id=cfg['camera_id'],
            position=cfg['position'],
            rotation=cfg['rotation'],
            fov=cfg['view_angle'],
            resolution=cfg['resolution'],
        ))
    return cams


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=int, default=3000,
                         help='Number of simulation ticks to run')
    parser.add_argument('--capture-every', type=int, default=5,
                         help='Save 1 image per camera every N ticks')
    parser.add_argument('--config', default='config/camera_config.yaml',
                         help='Camera config (defines CAM_001/002/003 viewpoints)')
    parser.add_argument('--num-vehicles', type=int, default=15)
    parser.add_argument('--num-pedestrians', type=int, default=8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='data/carla_cam_det')
    parser.add_argument('--yaml', default='data/carla_cam_det.yaml')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out)
    for split in ('train', 'val'):
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    _, world = connect_carla()

    traffic = TrafficGenerator(world, num_vehicles=args.num_vehicles,
                                num_pedestrians=args.num_pedestrians)
    traffic.spawn_actors()

    cam_configs = load_cam_configs(args.config)

    cameras = []
    try:
        for cfg in cam_configs:
            cam = CctvCamera(world, cfg['position'], cfg['rotation'], cfg['fov'],
                              cfg['resolution'], camera_id=cfg['camera_id'])
            cameras.append(cam)

        n_images, n_boxes = 0, 0
        for tick in range(args.frames):
            try:
                world.tick()
            except RuntimeError as e:
                logger.error("world.tick() failed at tick %d (%s) — stopping early", tick, e)
                break
            traffic.update_pedestrians()
            snapshot = world.get_snapshot()
            frame_id = snapshot.frame

            if tick % args.capture_every != 0:
                continue

            actors = world.get_actors()
            vehicles = list(actors.filter('vehicle.*'))
            walkers = list(actors.filter('walker.pedestrian.*'))

            split = 'val' if rng.random() < args.val_ratio else 'train'

            for cam in cameras:
                rgb = cam.get_frame(frame_id)
                if rgb is None:
                    continue
                width, height = cam.resolution

                lines = []
                for actor in vehicles:
                    box = project_bbox(cam, actor)
                    if box is None:
                        continue
                    cls = classify_vehicle(actor)
                    cx, cy, w, h = to_yolo_label(box, width, height)
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    n_boxes += 1

                for actor in walkers:
                    box = project_bbox(cam, actor)
                    if box is None:
                        continue
                    cx, cy, w, h = to_yolo_label(box, width, height)
                    lines.append(f"{VN_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    n_boxes += 1

                if not lines:
                    continue

                stem = f"carla_{frame_id:08d}_{cam.camera_id}"
                img_path = out_root / 'images' / split / f"{stem}.jpg"
                lbl_path = out_root / 'labels' / split / f"{stem}.txt"
                cv2.imwrite(str(img_path), rgb)
                lbl_path.write_text('\n'.join(lines) + '\n')
                n_images += 1

            if tick % (args.capture_every * 20) == 0:
                logger.info("tick %d/%d — %d images, %d boxes so far",
                             tick, args.frames, n_images, n_boxes)

    finally:
        for cam in cameras:
            cam.destroy()
        traffic.cleanup()

    write_yaml(out_root, Path(args.yaml))
    logger.info("Done: %d images, %d boxes -> %s", n_images, n_boxes, out_root)
    logger.info("Dataset config written to %s", args.yaml)


if __name__ == '__main__':
    main()
