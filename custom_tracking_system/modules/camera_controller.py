"""
Camera Controller Module
Handles camera setup, synchronization, and data collection in CARLA
"""

import carla
import numpy as np
from collections import deque
import yaml
import logging

logger = logging.getLogger(__name__)

class CameraController:
    """
    Manages multiple CCTV cameras in CARLA simulation
    """

    def __init__(self, client, world, config_path):
        """
        Initialize camera controller

        Args:
            client: CARLA client
            world: CARLA world
            config_path: Path to camera configuration YAML
        """
        self.client = client
        self.world = world
        self.cameras = {}
        self.sensor_data = {}
        self.config = None

        self.load_config(config_path)
        logger.info(f"CameraController initialized with config: {config_path}")

    def load_config(self, config_path):
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def setup_cameras(self):
        """
        Create camera sensors at specified positions
        """
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        for cam_name, cam_cfg in self.config['cameras'].items():
            # Configure camera blueprint
            camera_bp.set_attribute('image_size_x', str(cam_cfg['resolution'][0]))
            camera_bp.set_attribute('image_size_y', str(cam_cfg['resolution'][1]))
            camera_bp.set_attribute('fov', str(cam_cfg['view_angle']))

            # Set camera transform
            location = carla.Location(*cam_cfg['position'])
            rotation = carla.Rotation(*cam_cfg['rotation'])
            transform = carla.Transform(location, rotation)

            # Spawn camera actor
            camera = self.world.spawn_actor(camera_bp, transform)
            camera_id = cam_cfg['camera_id']

            # Setup data buffer
            self.cameras[camera_id] = camera
            self.sensor_data[camera_id] = deque(maxlen=100)

            # Listen for camera data
            camera.listen(lambda img, cam_id=camera_id:
                         self._store_image(img, cam_id))

            logger.info(f"Camera {camera_id} spawned at {cam_cfg['position']}")

    def _store_image(self, image, camera_id):
        """
        Store image data from camera sensor

        Args:
            image: CARLA image object
            camera_id: Camera identifier
        """
        try:
            # Convert CARLA image to numpy array
            data = {
                'timestamp': image.timestamp,
                'frame': np.array(image.raw_data).reshape(
                    image.height, image.width, 4)[:, :, :3],  # RGB only
                'camera_id': camera_id,
                'frame_number': image.frame
            }
            self.sensor_data[camera_id].append(data)
        except Exception as e:
            logger.error(f"Error storing image from {camera_id}: {e}")

    def get_synchronized_frames(self):
        """
        Get synchronized frames from all cameras

        Returns:
            dict: {camera_id: frame_data}
        """
        synchronized_data = {}

        # Get latest frame from each camera
        for cam_id, data_buffer in self.sensor_data.items():
            if len(data_buffer) > 0:
                synchronized_data[cam_id] = data_buffer[-1]

        return synchronized_data

    def get_camera_info(self, camera_id):
        """
        Get information about a specific camera

        Args:
            camera_id: Camera identifier

        Returns:
            dict: Camera configuration
        """
        for cam_name, cam_cfg in self.config['cameras'].items():
            if cam_cfg['camera_id'] == camera_id:
                return cam_cfg
        return None

    def cleanup(self):
        """Destroy all camera sensors"""
        for camera in self.cameras.values():
            try:
                camera.destroy()
                logger.info("Camera destroyed")
            except Exception as e:
                logger.error(f"Error destroying camera: {e}")

        self.cameras.clear()
        self.sensor_data.clear()