"""
Ground Truth Collector — TACH BIET khoi AI pipeline.

Module nay chi dung de DANH GIA do chinh xac, KHONG duoc su dung
trong qua trinh nhan dien/tracking.

Trong CARLA: doc actor_id, location, velocity tu CARLA World.
Trong thuc te: doc tu file annotation (MOT format, COCO format).

AI pipeline (detection/tracking/ReID/trajectory/alert logic) KHONG DUOC dung
du lieu tu module nay de ra quyet dinh.

Ngoai le: main.py (--source carla) duoc phep IMPORT va GOI
`project_actors_to_cameras()` CHI de ghi file gt_<CAM_ID>.txt khi chay voi
--eval-output (evaluation logging). Day la nhanh code rieng, khong anh huong
toi detection/tracking/global_id/alert/incident.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def project_actors_to_cameras(world, calibrations):
    """Project every vehicle/pedestrian 3D bounding box onto each camera's
    pixel space, for the CURRENT carla world tick.

    Used by carla_bridge/server.py (Python 3.7, has CARLA world access) to
    export ground-truth detections for MOT evaluation. NOT imported by the
    AI pipeline (main.py).

    Args:
        world: carla.World
        calibrations: {camera_id: CameraCalibration} (from CalibrationStore)

    Returns:
        {camera_id: [{'id': actor_id, 'box': [x1,y1,x2,y2], 'class': 'vehicle'|'pedestrian'}]}
    """
    actors = world.get_actors()
    targets = []
    for v in actors.filter('vehicle.*'):
        targets.append((v, 'vehicle'))
    for w in actors.filter('walker.pedestrian.*'):
        targets.append((w, 'pedestrian'))

    result = {cam_id: [] for cam_id in calibrations}

    for actor, cls in targets:
        try:
            verts = actor.bounding_box.get_world_vertices(actor.get_transform())
        except Exception:
            continue

        for cam_id, calib in calibrations.items():
            pts = []
            for v in verts:
                p = calib.project_point([v.x, v.y, v.z])
                if p is not None:
                    pts.append(p)
            if len(pts) < 2:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            # CARLA cameras: principal point == image center -> K[0,2], K[1,2]
            # give half-width/half-height, i.e. the image resolution.
            width, height = calib.K[0, 2] * 2, calib.K[1, 2] * 2

            x1c, y1c = max(0.0, x1), max(0.0, y1)
            x2c, y2c = min(width, x2), min(height, y2)
            if x2c - x1c < 2 or y2c - y1c < 2:
                continue

            result[cam_id].append({
                'id': actor.id,
                'box': [x1c, y1c, x2c, y2c],
                'class': cls,
            })

    return result


class GroundTruthCollector:
    """Thu thap ground truth tu CARLA World.

    QUAN TRONG: Module nay nam NGOAI AI pipeline.
    AI pipeline khong biet va khong duoc phep truy cap thong tin nay.
    Chi dung cho evaluation/metrics.
    """

    def __init__(self, carla_world):
        self._world = carla_world
        self._history = []  # List of frame snapshots
        logger.info("GroundTruthCollector initialized (evaluation only)")

    def collect_frame(self, frame_number: int) -> dict:
        """Thu thap ground truth cua tat ca actors tai 1 thoi diem.

        Returns:
            dict: {
                'frame': int,
                'timestamp': str,
                'vehicles': [
                    {
                        'actor_id': int,       # ID that cua CARLA
                        'type_id': str,        # "vehicle.tesla.model3"
                        'location': [x, y, z],
                        'velocity': [vx, vy, vz],
                        'bounding_box': {center, extent},
                    }
                ],
                'pedestrians': [...]
            }
        """
        actors = self._world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')

        frame_data = {
            'frame': frame_number,
            'timestamp': datetime.utcnow().isoformat(),
            'vehicles': [],
            'pedestrians': [],
        }

        for v in vehicles:
            loc = v.get_location()
            vel = v.get_velocity()
            bb = v.bounding_box
            frame_data['vehicles'].append({
                'actor_id': v.id,
                'type_id': v.type_id,
                'location': [loc.x, loc.y, loc.z],
                'velocity': [vel.x, vel.y, vel.z],
                'bounding_box': {
                    'center': [bb.location.x, bb.location.y, bb.location.z],
                    'extent': [bb.extent.x, bb.extent.y, bb.extent.z],
                },
            })

        for w in walkers:
            loc = w.get_location()
            vel = w.get_velocity()
            frame_data['pedestrians'].append({
                'actor_id': w.id,
                'type_id': w.type_id,
                'location': [loc.x, loc.y, loc.z],
                'velocity': [vel.x, vel.y, vel.z],
            })

        self._history.append(frame_data)
        return frame_data

    def get_actor_id_at_location(self, location_2d: list, camera_transform,
                                  camera_intrinsics: dict) -> Optional[int]:
        """Map 1 diem 2D tren camera frame ve actor_id (ground truth).

        Dung cho evaluation: khi AI pipeline noi "day la Global ID 1042",
        ta project nguoc ve 3D de xem CARLA actor_id that su la gi,
        roi so sanh.
        """
        # TODO: implement 2D -> 3D projection + nearest actor matching
        pass

    def export_mot_format(self, output_path: str):
        """Export ground truth theo MOT Challenge format.

        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
        Dung voi py-motmetrics de tinh MOTA, IDF1.
        """
        # TODO: implement
        pass

    def get_history(self) -> list:
        return self._history

    def clear(self):
        self._history.clear()


class FileGroundTruth:
    """Doc ground truth tu file annotation (cho video that / dataset).

    Supported formats:
      - MOT Challenge format (gt.txt)
      - COCO format (annotations.json)
    """

    def __init__(self, filepath: str, format: str = "mot"):
        self._filepath = filepath
        self._format = format
        self._data = {}  # {frame_number: [annotations]}
        self._load()

    def _load(self):
        if self._format == "mot":
            self._load_mot()
        else:
            logger.error(f"Unsupported format: {self._format}")

    def _load_mot(self):
        """Load MOT Challenge format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        try:
            with open(self._filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    frame = int(parts[0])
                    obj_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                    if frame not in self._data:
                        self._data[frame] = []
                    self._data[frame].append({
                        'id': obj_id,
                        'box': [int(x), int(y), int(x + w), int(y + h)],
                    })
            logger.info(f"Loaded MOT ground truth: {len(self._data)} frames")
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")

    def get_frame(self, frame_number: int) -> list:
        """Lay annotations cho 1 frame.

        Returns:
            list: [{'id': int, 'box': [x1,y1,x2,y2]}]
        """
        return self._data.get(frame_number, [])
