"""
Data Writer Module
Handles saving tracking data, trajectories, and results
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataWriter:
    """
    Saves tracking data to various formats
    """

    def __init__(self, output_dir="output"):
        """
        Initialize data writer

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.trajectories_dir = self.output_dir / "trajectories"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.trajectories_dir, self.snapshots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        logger.info(f"DataWriter initialized with output dir: {output_dir}")

    def save_trajectory(self, global_id, trajectory_data, camera_id=None):
        """
        Save trajectory data for an object

        Args:
            global_id: Global object ID
            trajectory_data: dict with 'positions' and 'frames'
            camera_id: Camera identifier (optional)
        """
        try:
            filename = f"trajectory_{global_id}"
            if camera_id:
                filename += f"_{camera_id}"
            filename += ".json"

            filepath = self.trajectories_dir / filename

            # Convert to serializable format
            data = {
                'global_id': global_id,
                'camera_id': camera_id,
                'timestamp': datetime.now().isoformat(),
                'positions': trajectory_data.get('positions', []),
                'frames': trajectory_data.get('frames', [])
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved trajectory for ID {global_id}")

        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")

    def save_global_tracks_snapshot(self, frame_count, global_tracks):
        """
        Save snapshot of current global tracks

        Args:
            frame_count: Current frame number
            global_tracks: List of current global tracks
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracks_snapshot_{frame_count}_{timestamp}.json"
            filepath = self.snapshots_dir / filename

            data = {
                'frame_count': frame_count,
                'timestamp': datetime.now().isoformat(),
                'tracks': global_tracks
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved tracks snapshot at frame {frame_count}")

        except Exception as e:
            logger.error(f"Failed to save tracks snapshot: {e}")

    def save_alert_log(self, alert):
        """
        Save alert to log file

        Args:
            alert: Alert dictionary
        """
        try:
            filename = f"alerts_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.logs_dir / filename

            # Check if file exists to write header
            file_exists = filepath.exists()

            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(['timestamp', 'global_id', 'camera_id',
                                   'roi_name', 'type', 'eta_frames'])

                writer.writerow([
                    alert['timestamp'].isoformat(),
                    alert['global_id'],
                    alert['camera_id'],
                    alert['roi_name'],
                    alert['type'],
                    alert['eta_frames']
                ])

            logger.debug(f"Logged alert for ID {alert['global_id']}")

        except Exception as e:
            logger.error(f"Failed to save alert log: {e}")

    def save_system_log(self, log_data):
        """
        Save system performance log

        Args:
            log_data: Dictionary of system metrics
        """
        try:
            filename = f"system_log_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.logs_dir / filename

            file_exists = filepath.exists()

            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(['timestamp'] + list(log_data.keys()))

                row = [datetime.now().isoformat()] + list(log_data.values())
                writer.writerow(row)

            logger.debug("Saved system log entry")

        except Exception as e:
            logger.error(f"Failed to save system log: {e}")

    def export_all_trajectories(self, trajectories_dict, filename=None):
        """
        Export all trajectories to a single file

        Args:
            trajectories_dict: Dict of {global_id: trajectory_data}
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_trajectories_{timestamp}.json"

        filepath = self.output_dir / filename

        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'trajectories': trajectories_dict
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported all trajectories to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export trajectories: {e}")

    def save_camera_frame(self, camera_id, frame, frame_count):
        """
        Save camera frame as image

        Args:
            camera_id: Camera identifier
            frame: numpy array
            frame_count: Frame number
        """
        try:
            import cv2

            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"frame_{camera_id}_{frame_count}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename

            cv2.imwrite(str(filepath), frame)
            logger.debug(f"Saved frame {frame_count} from camera {camera_id}")

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")

    def create_summary_report(self, metrics_summary):
        """
        Create a summary report of the tracking session

        Args:
            metrics_summary: Metrics summary dictionary
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.txt"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                f.write("MULTI-CAMERA TRACKING SYSTEM REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")

                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Runtime: {metrics_summary.get('runtime_seconds', 0):.2f} seconds\n")
                f.write(f"Total Frames: {metrics_summary.get('total_frames', 0)}\n")
                f.write(f"Average FPS: {metrics_summary.get('average_fps', 0):.2f}\n\n")

                f.write("DETECTION:\n")
                f.write("-" * 10 + "\n")
                det = metrics_summary.get('detection', {})
                f.write(f"Total Detections: {det.get('total_detections', 0)}\n")
                f.write(f"Avg Detections/Frame: {det.get('avg_detections_per_frame', 0):.2f}\n")
                f.write(f"Avg Detection Time: {det.get('avg_detection_time', 0):.4f}s\n\n")

                f.write("TRACKING:\n")
                f.write("-" * 9 + "\n")
                track = metrics_summary.get('tracking', {})
                f.write(f"Total Tracks: {track.get('total_tracks', 0)}\n")
                f.write(f"Avg Tracks/Frame: {track.get('avg_tracks_per_frame', 0):.2f}\n")
                f.write(f"Unique Global IDs: {track.get('unique_global_ids', 0)}\n\n")

                f.write("RE-IDENTIFICATION:\n")
                f.write("-" * 18 + "\n")
                reid = metrics_summary.get('reid', {})
                f.write(f"Total Attempts: {reid.get('total_attempts', 0)}\n")
                f.write(f"Successful Matches: {reid.get('successful_matches', 0)}\n")
                f.write(f"Match Rate: {reid.get('match_rate', 0):.1%}\n\n")

                f.write("CROSS-CAMERA TRANSFERS:\n")
                f.write("-" * 22 + "\n")
                cc = metrics_summary.get('cross_camera', {})
                f.write(f"Total Transfers: {cc.get('transfers', 0)}\n")

            logger.info(f"Summary report saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")

    def get_output_files(self):
        """
        Get list of all output files

        Returns:
            dict: categorized file lists
        """
        files = {
            'trajectories': list(self.trajectories_dir.glob("*.json")),
            'snapshots': list(self.snapshots_dir.glob("*")),
            'logs': list(self.logs_dir.glob("*")),
            'reports': list(self.output_dir.glob("*.txt"))
        }

        return files

    def cleanup_old_files(self, days_to_keep=7):
        """
        Remove old output files

        Args:
            days_to_keep: Number of days of files to keep
        """
        try:
            import time
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

            for dir_path in [self.trajectories_dir, self.snapshots_dir, self.logs_dir]:
                for file_path in dir_path.glob("*"):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        logger.debug(f"Removed old file: {file_path}")

            logger.info(f"Cleaned up files older than {days_to_keep} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")