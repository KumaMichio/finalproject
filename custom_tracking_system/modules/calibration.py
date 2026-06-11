"""
Camera Calibration Module
Pixel <-> World (ground-plane, meters) conversion per camera.

Two ways to build a CameraCalibration:

1. CARLA / simulation — `from_carla_params(position, rotation, fov, resolution)`
   Uses the camera's known intrinsics (fov + resolution) and extrinsics
   (CARLA world position/rotation) to build a full projective camera model.
   `pixel_to_world()` ray-casts from the camera center through the pixel and
   intersects the world ground plane (Z = ground_z, default 0 — CARLA road
   surface).

2. Real CCTV — `from_point_correspondences(pixel_pts, world_pts)`
   Manual calibration: give >= 4 (pixel_xy, world_xy) correspondences measured
   on site (e.g. lane markings with known real-world spacing) and a
   ground-plane homography is fitted with cv2.findHomography. Not the focus
   right now, but kept behind the same interface so the rest of the pipeline
   (TrajectoryPredictor ensemble) does not need to know which mode is active.

Both modes expose the same public interface:
    pixel_to_world(u, v) -> (x, y)              # meters, world frame
    world_to_pixel(x, y, z=0.0) -> (u, v) | None # None if behind camera (projective mode only)
    project_point(world_xyz) -> (u, v) | None    # alias of world_to_pixel for a 3D point
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# CARLA camera local axes (X-forward, Y-right, Z-up) -> standard CV camera
# axes (x-right, y-down, z-forward). Fixed rotation, same for every camera.
_CARLA_TO_CV = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def _carla_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """3x3 rotation matrix mapping camera-local axes -> world axes (CARLA convention).

    Equivalent to the rotation part of carla.Transform.get_matrix().
    """
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))

    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,                cp * cr],
    ])


class CameraCalibration:
    """Pixel <-> world (ground-plane, meters) conversion for a single camera."""

    def __init__(self):
        self._mode = None  # 'projective' | 'homography'

        # Projective mode state
        self.K = None            # 3x3 intrinsic matrix
        self.R_c2w = None        # 3x3 rotation, camera-local -> world
        self.cam_pos = None      # (3,) camera position in world (meters)
        self.ground_z = 0.0      # world Z of the ground plane

        # Homography mode state
        self.H = None            # pixel -> world (3x3)
        self.H_inv = None        # world -> pixel (3x3)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_carla_params(cls, position, rotation, fov, resolution, ground_z: float = 0.0):
        """Build a projective calibration from CARLA camera config.

        Args:
            position:   [x, y, z] world position of the camera (meters).
            rotation:   [pitch, yaw, roll] in degrees (CARLA convention).
            fov:        horizontal field of view, degrees.
            resolution: [width, height] in pixels.
            ground_z:   world Z of the ground plane actors move on.
        """
        calib = cls()
        calib._mode = 'projective'

        width, height = resolution
        focal = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        calib.K = np.array([
            [focal, 0.0,   width / 2.0],
            [0.0,   focal, height / 2.0],
            [0.0,   0.0,   1.0],
        ])

        pitch, yaw, roll = rotation
        calib.R_c2w = _carla_rotation_matrix(pitch, yaw, roll)
        calib.cam_pos = np.array(position, dtype=float)
        calib.ground_z = ground_z

        return calib

    @classmethod
    def from_point_correspondences(cls, pixel_pts, world_pts):
        """Build a homography-based calibration from manual correspondences.

        Args:
            pixel_pts: Nx2 array-like of (u, v) pixel coordinates, N >= 4.
            world_pts: Nx2 array-like of (x, y) world coordinates (meters), N >= 4.
        """
        import cv2

        pixel_pts = np.asarray(pixel_pts, dtype=np.float32)
        world_pts = np.asarray(world_pts, dtype=np.float32)
        if len(pixel_pts) < 4 or len(world_pts) < 4:
            raise ValueError("Need at least 4 point correspondences for homography calibration")

        H, _ = cv2.findHomography(pixel_pts, world_pts, method=0)
        if H is None:
            raise ValueError("cv2.findHomography failed to compute a homography")

        calib = cls()
        calib._mode = 'homography'
        calib.H = H
        calib.H_inv = np.linalg.inv(H)
        return calib

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pixel_to_world(self, u: float, v: float):
        """Project a pixel (u, v) onto the ground plane -> (x, y) world meters."""
        if self._mode == 'homography':
            return self._apply_homography(self.H, u, v)
        return self._pixel_to_world_projective(u, v)

    def world_to_pixel(self, x: float, y: float, z: float = 0.0):
        """Project a ground-plane world point (x, y[, z]) -> (u, v) pixels.

        Returns None if the point is behind the camera (projective mode only).
        """
        if self._mode == 'homography':
            return self._apply_homography(self.H_inv, x, y)
        return self._world_to_pixel_projective(np.array([x, y, z], dtype=float))

    def project_point(self, world_xyz):
        """Project an arbitrary 3D world point -> (u, v) pixels, or None if behind camera."""
        if self._mode == 'homography':
            x, y = world_xyz[0], world_xyz[1]
            return self._apply_homography(self.H_inv, x, y)
        return self._world_to_pixel_projective(np.asarray(world_xyz, dtype=float))

    # ------------------------------------------------------------------
    # Projective-mode internals (CARLA)
    # ------------------------------------------------------------------

    def _world_to_pixel_projective(self, world_xyz: np.ndarray):
        rel = world_xyz - self.cam_pos
        cam_local = self.R_c2w.T @ rel          # world -> camera-local (CARLA axes)
        cv_coords = _CARLA_TO_CV @ cam_local    # CARLA cam axes -> CV axes (x-right, y-down, z-fwd)

        if cv_coords[2] <= 1e-6:
            return None  # behind (or on) the camera plane

        pixel = self.K @ cv_coords
        u = pixel[0] / pixel[2]
        v = pixel[1] / pixel[2]
        return (float(u), float(v))

    def _pixel_to_world_projective(self, u: float, v: float):
        # Ray direction in CV camera axes
        dir_cv = np.linalg.solve(self.K, np.array([u, v, 1.0]))
        # CV axes -> CARLA camera-local axes (rotation matrices: inverse == transpose)
        dir_cam_local = _CARLA_TO_CV.T @ dir_cv
        # Camera-local -> world
        dir_world = self.R_c2w @ dir_cam_local

        if abs(dir_world[2]) < 1e-9:
            logger.warning("pixel_to_world: ray parallel to ground plane, cannot intersect")
            return (float(self.cam_pos[0]), float(self.cam_pos[1]))

        s = (self.ground_z - self.cam_pos[2]) / dir_world[2]
        x = self.cam_pos[0] + s * dir_world[0]
        y = self.cam_pos[1] + s * dir_world[1]
        return (float(x), float(y))

    # ------------------------------------------------------------------
    # Homography-mode internals (real CCTV)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_homography(matrix: np.ndarray, x: float, y: float):
        vec = matrix @ np.array([x, y, 1.0])
        if abs(vec[2]) < 1e-12:
            return None
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


class CalibrationStore:
    """Loads CameraCalibration instances for every camera in camera_config.yaml."""

    @staticmethod
    def load_from_config(config_path: str, ground_z: float = 0.0) -> dict:
        """
        Args:
            config_path: path to camera_config.yaml.
            ground_z:    world Z of the ground plane (CARLA road surface ~= 0).

        Returns:
            {camera_id: CameraCalibration}
        """
        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        calibrations = {}
        for _, cam_cfg in (config.get('cameras') or {}).items():
            camera_id = cam_cfg['camera_id']
            try:
                calibrations[camera_id] = CameraCalibration.from_carla_params(
                    position=cam_cfg['position'],
                    rotation=cam_cfg['rotation'],
                    fov=cam_cfg['view_angle'],
                    resolution=cam_cfg['resolution'],
                    ground_z=ground_z,
                )
            except KeyError as e:
                logger.warning("Skipping calibration for %s: missing field %s", camera_id, e)

        logger.info("CalibrationStore: loaded %d camera calibration(s) from %s",
                     len(calibrations), config_path)
        return calibrations
