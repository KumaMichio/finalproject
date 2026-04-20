#!/usr/bin/env python3
"""
Multi-Camera CCTV Tracking System for CARLA
Main entry point for the tracking system
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.camera_controller import CameraController
from modules.traffic_generator import TrafficGenerator
from modules.detector import ObjectDetector
from modules.tracker import SimpleTracker
from modules.reid import ReIDExtractor
from modules.global_tracking import GlobalTracker
from modules.trajectory_predictor import TrajectoryPredictor
from modules.alert_system import AlertSystem
from utils.visualization import Visualizer
from utils.metrics import MetricsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrackingSystem:
    def __init__(self, config_path):
        self.config_path = config_path
        self.carla_client = None
        self.world = None
        self.modules = {}

    def initialize_carla(self):
        """Initialize CARLA client and world"""
        try:
            import carla
            self.carla_client = carla.Client('localhost', 2000)
            self.carla_client.set_timeout(10.0)
            self.world = self.carla_client.get_world()
            logger.info("CARLA client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CARLA: {e}")
            return False

    def setup_synchronous_mode(self):
        """Enable synchronous mode for deterministic simulation"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        logger.info("Synchronous mode enabled")

    def initialize_modules(self):
        """Initialize all tracking modules"""
        def step(name):
            print(f"[INIT] {name}...", flush=True)
            logger.info(f"Initializing: {name}")

        try:
            step("CameraController")
            self.modules['camera_controller'] = CameraController(
                self.carla_client, self.world, self.config_path)
            self.modules['camera_controller'].setup_cameras()

            step("TrafficGenerator")
            self.modules['traffic_generator'] = TrafficGenerator(
                self.world, num_vehicles=10, num_pedestrians=5)
            self.modules['traffic_generator'].spawn_actors()

            step("ObjectDetector")
            self.modules['detector'] = ObjectDetector(model_type='yolov5s')

            step("SimpleTracker (per camera)")
            self.modules['trackers'] = {}
            for cam_id in self.modules['camera_controller'].cameras.keys():
                self.modules['trackers'][cam_id] = SimpleTracker(max_age=30, min_hits=3)

            step("ReIDExtractor")
            self.modules['reid_extractor'] = ReIDExtractor(model_name='resnet50')

            step("GlobalTracker")
            self.modules['global_tracker'] = GlobalTracker(self.modules['reid_extractor'])

            step("TrajectoryPredictor")
            self.modules['trajectory_predictor'] = TrajectoryPredictor(window_size=10, pred_steps=5)

            step("AlertSystem")
            self.modules['alert_system'] = AlertSystem(self.modules['trajectory_predictor'])
            self.modules['alert_system'].set_rois(self.modules['camera_controller'].config['rois'])

            step("Visualizer")
            self.modules['visualizer'] = Visualizer()

            step("MetricsCollector")
            self.modules['metrics'] = MetricsCollector()

            print("[INIT] All modules OK", flush=True)
            logger.info("All modules initialized successfully")
            return True

        except Exception as e:
            print(f"[INIT ERROR] {e}", flush=True)
            logger.error(f"Failed to initialize modules: {e}", exc_info=True)
            return False

    def run(self, max_frames=None):
        """Main processing loop"""
        frame_count = 0
        logger.info("Starting tracking system...")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached maximum frames ({max_frames}), stopping...")
                    break

                frame_count += 1

                # Get synchronized frames
                sync_frames = self.modules['camera_controller'].get_synchronized_frames()

                if not sync_frames:
                    logger.warning("No frames received, skipping...")
                    self.world.tick()
                    continue

                all_global_tracks = []

                # Process each camera
                for camera_id, frame_data in sync_frames.items():
                    frame = frame_data['frame']

                    # Detection
                    detections = self.modules['detector'].detect(frame)

                    # Single-camera tracking
                    local_tracks = self.modules['trackers'][camera_id].update(detections)

                    # Global tracking (cross-camera)
                    global_tracks = self.modules['global_tracker'].process_camera_tracks(
                        camera_id, frame, local_tracks)

                    all_global_tracks.extend(global_tracks)

                    # Update trajectories
                    for g_track in global_tracks:
                        global_id = g_track['global_id']
                        box = g_track['box']
                        center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                        self.modules['trajectory_predictor'].update_trajectory(
                            global_id, center, frame_count)

                    # Trajectory prediction & alerts
                    for g_track in global_tracks:
                        global_id = g_track['global_id']
                        predicted = self.modules['trajectory_predictor'].predict(global_id)

                        if predicted is not None:
                            alerts = self.modules['alert_system'].check_alerts(
                                global_id, camera_id, predicted, g_track['box'])

                            for alert in alerts:
                                self.modules['alert_system'].log_alert(alert)

                    # Visualization
                    vis_frame = self.modules['visualizer'].draw_tracks(frame, global_tracks)

                    # Display (optional - comment out for headless operation)
                    import cv2
                    cv2.imshow(f'{camera_id}', vis_frame)

                # Collect metrics
                self.modules['metrics'].update(frame_count, all_global_tracks)

                # Advance CARLA simulation
                self.world.tick()

                # Check for exit (optional)
                import cv2
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exit signal received, stopping...")
                    break

                # Log progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")

        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass

        if 'camera_controller' in self.modules:
            self.modules['camera_controller'].cleanup()

        if 'traffic_generator' in self.modules:
            self.modules['traffic_generator'].cleanup()

        # Disable synchronous mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        logger.info("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Multi-Camera CCTV Tracking System')
    parser.add_argument('--config', type=str, default='config/camera_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize system
    system = TrackingSystem(args.config)

    if not system.initialize_carla():
        sys.exit(1)

    system.setup_synchronous_mode()

    if not system.initialize_modules():
        sys.exit(1)

    # Run the system
    system.run(max_frames=args.max_frames)

if __name__ == '__main__':
    main()