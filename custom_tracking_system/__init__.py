# Multi-Camera CCTV Tracking System
# Main package initialization

__version__ = "1.0.0"
__author__ = "CARLA Tracking System"

# Import main modules for easy access
from .modules.camera_controller import CameraController
from .modules.traffic_generator import TrafficGenerator
from .modules.detector import ObjectDetector
from .modules.tracker import SimpleTracker
from .modules.reid import ReIDExtractor
from .modules.global_tracking import GlobalTracker
from .modules.trajectory_predictor import TrajectoryPredictor
from .modules.alert_system import AlertSystem

from .utils.visualization import Visualizer
from .utils.metrics import MetricsCollector
from .utils.data_writer import DataWriter

__all__ = [
    'CameraController',
    'TrafficGenerator',
    'ObjectDetector',
    'SimpleTracker',
    'ReIDExtractor',
    'GlobalTracker',
    'TrajectoryPredictor',
    'AlertSystem',
    'Visualizer',
    'MetricsCollector',
    'DataWriter'
]