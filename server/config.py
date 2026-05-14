"""
Server configuration.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_SYSTEM_DIR = PROJECT_ROOT / "custom_tracking_system"
CAMERA_CONFIG_PATH = TRACKING_SYSTEM_DIR / "config" / "camera_config.yaml"

# Server
HOST = "0.0.0.0"
PORT = 8000

# CARLA
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# AI Pipeline
DETECTOR_MODEL = "yolov5s"
DETECTOR_CONFIDENCE = 0.4
REID_THRESHOLD = 0.5
MAX_VEHICLES = 10
MAX_PEDESTRIANS = 5

# Streaming
JPEG_QUALITY = 70
STREAM_FPS = 10
