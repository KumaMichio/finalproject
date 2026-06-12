#!/usr/bin/env python3
"""
CARLA Object Detection Dataset Collector

Spawns traffic in CARLA and a number of elevated, wall-mounted "CCTV-style"
RGB cameras placed near street-corner junctions (mimicking real cameras
mounted on building corners overlooking an intersection). For every camera
frame, every visible vehicle/pedestrian is auto-labelled with a 2D bounding
box, projected from CARLA's 3D actor bounding boxes using the same
projective camera model as `modules/calibration.py`. A paired depth camera
is used to discard objects that are occluded behind closer geometry.

Output is a YOLO-format detection dataset using the same 5 VN classes as
`prepare_visdrone_dataset.py` (person, car, motorcycle, bus, truck), so it
can be combined with the VisDrone-VN dataset via
`merge_detection_datasets.py`.

Camera placement ("street corner, mounted on building wall"):
    For each road junction in the map, a camera is placed `dist` meters away
    from the junction center (dist = junction radius + 8-15m, i.e. roughly at
    the corner of a building facing the intersection), at height 5-8m, tilted
    down (pitch ~ -15 to -30 deg) to look at the junction center — the
    classic elevated CCTV angle.

Usage:
    # Run with CARLA server already running (Town10HD_Opt or similar)
    python collect_carla_detection_data.py --frames 2000 --num-cameras 4 \
        --num-vehicles 30 --num-pedestrians 15 --out data/carla_det

Output:
    data/carla_det/images/{train,val}/*.jpg
    data/carla_det/labels/{train,val}/*.txt
    data/carla_det.yaml   <- ultralytics dataset config (5 VN classes)

NOTE: written for CARLA 0.9.14 / Python 3.7 (`carla-sim` env, see
docs/execute.md). Not yet run end-to-end against a live CARLA server —
verify camera placement / bbox projection on a small `--frames 50` run
before doing a full collection.
"""

import argparse
import logging
import math
import queue
import random
import sys
from pathlib import Path

import cv2
import numpy as np

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

# Fallback heuristics for CARLA blueprints without a usable 'base_type' attribute
_MOTORCYCLE_HINTS = (
    'harley-davidson', 'kawasaki', 'yamaha', 'vespa',
    'crossbike', 'omafiets', 'century', 'gazelle', 'diamondback', 'bh',
)
_TRUCK_HINTS = ('carlacola', 'cybertruck', 'firetruck', 'truck')
_BUS_HINTS = ('fusorosa', 'sprinter')

# Minimum 2D bbox size (pixels) to keep a label
MIN_BOX_PX = 6
# Discard actors farther than this from the camera (bbox becomes tiny/noisy)
MAX_DISTANCE_M = 80.0
# Depth-buffer slack (meters) when checking center-point occlusion
DEPTH_SLACK_M = 0.5


# ---------------------------------------------------------------------------
# CARLA connection / actor classification
# ---------------------------------------------------------------------------

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
    """Map a CARLA vehicle actor to a VN class index (car/motorcycle/bus/truck)."""
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


# ---------------------------------------------------------------------------
# Camera placement — elevated, wall-mounted CCTV at street corners
# ---------------------------------------------------------------------------

def find_junction_centers(world, max_cameras: int, rng: random.Random):
    """Pick up to `max_cameras` distinct road junctions from the map topology."""
    import carla  # noqa: F401  (ensures carla is importable in this env)

    topology = world.get_map().get_topology()
    junctions = {}
    for wp1, wp2 in topology:
        for wp in (wp1, wp2):
            if wp.is_junction:
                j = wp.get_junction()
                if j is not None:
                    junctions[j.id] = j

    junction_list = list(junctions.values())
    rng.shuffle(junction_list)
    return junction_list[:max_cameras]


def camera_transform_for_junction(world, junction, rng: random.Random):
    """Place a camera at a 'building corner / pole' near the junction, looking down at it.

    To avoid cameras ending up underground or buried inside building meshes,
    the (x, y) position is snapped onto the road/sidewalk network near the
    junction corner via `world.get_map().get_waypoint()` (guaranteed open
    space), and the height is anchored to the actual terrain surface via
    `world.ground_projection()` plus a 5-8m pole/wall-mount offset.

    Returns (position [x,y,z], rotation [pitch, yaw, roll]).
    """
    import carla
    bb = junction.bounding_box
    center = bb.location
    radius = max(bb.extent.x, bb.extent.y, 5.0)

    angle_deg = rng.uniform(0, 360)
    dist = radius + rng.uniform(8.0, 15.0)

    angle_rad = math.radians(angle_deg)
    probe_x = center.x + dist * math.cos(angle_rad)
    probe_y = center.y + dist * math.sin(angle_rad)

    # Snap (x, y) onto the road/sidewalk network — keeps the camera base in
    # open space near the intersection corner, not inside a building footprint.
    waypoint = world.get_map().get_waypoint(
        carla.Location(probe_x, probe_y, center.z),
        project_to_road=True, lane_type=carla.LaneType.Any)
    base_loc = waypoint.transform.location

    # Anchor height to the actual ground/terrain surface under that point
    # (handles sloped maps — avoids floating above or sinking below ground).
    ground_hit = world.ground_projection(
        carla.Location(base_loc.x, base_loc.y, base_loc.z + 50.0), 100.0)
    ground_z = ground_hit.location.z if ground_hit is not None else base_loc.z

    height = rng.uniform(5.0, 8.0)  # pole / building-wall mount height
    cam_x, cam_y, cam_z = base_loc.x, base_loc.y, ground_z + height

    # Yaw/pitch: look down at the junction center, roughly at vehicle height
    target_z = ground_z + 1.5
    yaw = math.degrees(math.atan2(center.y - cam_y, center.x - cam_x))
    horiz_dist = math.hypot(center.x - cam_x, center.y - cam_y)
    pitch = -math.degrees(math.atan2(cam_z - target_z, max(horiz_dist, 1e-3)))
    pitch = max(min(pitch, -10.0), -35.0)  # clamp to a believable CCTV tilt

    return [cam_x, cam_y, cam_z], [pitch, yaw, 0.0]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

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

        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(resolution[0]))
        depth_bp.set_attribute('image_size_y', str(resolution[1]))
        depth_bp.set_attribute('fov', str(fov))
        self.depth_sensor = world.spawn_actor(depth_bp, transform)
        self.depth_queue = queue.Queue()
        self.depth_sensor.listen(self.depth_queue.put)

        logger.info("Camera %s: pos=%s rot=%s fov=%.0f res=%s",
                     camera_id, position, rotation, fov, resolution)

    def get_frame(self, frame_id, timeout=2.0):
        """Return (rgb_bgr, depth_meters) for the given simulation frame, or None."""
        rgb_img = self._pop_until(self.rgb_queue, frame_id, timeout)
        depth_img = self._pop_until(self.depth_queue, frame_id, timeout)
        if rgb_img is None or depth_img is None:
            return None

        w, h = self.resolution
        rgb = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]

        depth_raw = np.frombuffer(depth_img.raw_data, dtype=np.uint8).reshape((h, w, 4)).astype(np.float32)
        # CARLA depth encoding (BGRA buffer): depth_m = (R + G*256 + B*256^2) / (256^3-1) * 1000
        normalized = (depth_raw[:, :, 2] + depth_raw[:, :, 1] * 256.0 + depth_raw[:, :, 0] * 256.0 * 256.0) \
            / (256.0 ** 3 - 1.0)
        depth_m = normalized * 1000.0

        return rgb, depth_m

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
        for sensor in (self.rgb_sensor, self.depth_sensor):
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bounding box projection
# ---------------------------------------------------------------------------

def actor_world_corners(actor):
    """8 world-space corners of an actor's 3D bounding box."""
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


def project_bbox(camera: CctvCamera, actor, depth_m: np.ndarray):
    """Project an actor's 3D bbox onto the camera image -> (x1,y1,x2,y2) or None."""
    width, height = camera.resolution
    corners = actor_world_corners(actor)

    px = []
    for wx, wy, wz in corners:
        proj = camera.calib.project_point((wx, wy, wz))
        if proj is None:
            continue
        px.append(proj)

    # Need at least a few corners in front of the camera
    if len(px) < 4:
        return None

    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))

    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None

    # Occlusion check: compare depth buffer at bbox center vs actual distance
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cx = min(max(cx, 0), width - 1)
    cy = min(max(cy, 0), height - 1)

    actor_loc = actor.get_transform().location
    cam_pos = camera.calib.cam_pos
    distance = math.sqrt((actor_loc.x - cam_pos[0]) ** 2
                          + (actor_loc.y - cam_pos[1]) ** 2
                          + (actor_loc.z - cam_pos[2]) ** 2)
    if distance > MAX_DISTANCE_M:
        return None

    buffer_depth = float(depth_m[cy, cx])
    if buffer_depth + DEPTH_SLACK_M < distance:
        return None  # something closer is occluding the bbox center

    return x1, y1, x2, y2


def to_yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=int, default=2000,
                         help='Number of simulation frames to capture across all cameras')
    parser.add_argument('--capture-every', type=int, default=5,
                         help='Save 1 image per camera every N ticks (avoids near-duplicate frames)')
    parser.add_argument('--num-cameras', type=int, default=4)
    parser.add_argument('--num-vehicles', type=int, default=30)
    parser.add_argument('--num-pedestrians', type=int, default=15)
    parser.add_argument('--resolution', type=int, nargs=2, default=[960, 540])
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='data/carla_det')
    parser.add_argument('--yaml', default='data/carla_det.yaml')
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

    junctions = find_junction_centers(world, args.num_cameras, rng)
    if not junctions:
        logger.warning("No junctions found in map topology; falling back to map spawn points")

    cameras = []
    try:
        for i in range(args.num_cameras):
            if junctions:
                junction = junctions[i % len(junctions)]
                position, rotation = camera_transform_for_junction(world, junction, rng)
            else:
                sp = world.get_map().get_spawn_points()[i % len(world.get_map().get_spawn_points())]
                position = [sp.location.x, sp.location.y, sp.location.z + 6.0]
                rotation = [-20.0, sp.rotation.yaw, 0.0]

            cam = CctvCamera(world, position, rotation, args.fov, args.resolution,
                              camera_id=f"CAM_{i+1:03d}")
            cameras.append(cam)

        n_images, n_boxes = 0, 0
        for tick in range(args.frames):
            world.tick()
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
                result = cam.get_frame(frame_id)
                if result is None:
                    continue
                rgb, depth_m = result
                width, height = cam.resolution

                lines = []
                for actor in vehicles:
                    box = project_bbox(cam, actor, depth_m)
                    if box is None:
                        continue
                    cls = classify_vehicle(actor)
                    cx, cy, w, h = to_yolo_label(box, width, height)
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    n_boxes += 1

                for actor in walkers:
                    box = project_bbox(cam, actor, depth_m)
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
