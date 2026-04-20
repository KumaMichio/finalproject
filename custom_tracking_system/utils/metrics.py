"""
Metrics Collection Module
Collects and analyzes tracking performance metrics
"""

import numpy as np
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects tracking metrics and performance statistics
    """

    def __init__(self):
        self.reset()
        logger.info("MetricsCollector initialized")

    def reset(self):
        """Reset all metrics"""
        self.frame_count = 0
        self.start_time = time.time()

        # Detection metrics
        self.detection_counts = defaultdict(int)
        self.detection_times = []

        # Tracking metrics
        self.track_counts = defaultdict(int)
        self.tracking_times = []

        # ReID metrics
        self.reid_matches = 0
        self.reid_attempts = 0
        self.reid_times = []

        # Global tracking metrics
        self.global_id_counts = defaultdict(int)
        self.cross_camera_transfers = 0

        # Performance metrics
        self.fps_history = []
        self.memory_usage = []

    def update(self, frame_count, global_tracks):
        """
        Update metrics for current frame

        Args:
            frame_count: Current frame number
            global_tracks: List of current global tracks
        """
        self.frame_count = frame_count

        # Count objects by class
        class_counts = defaultdict(int)
        for track in global_tracks:
            class_name = track.get('class', 'unknown')
            class_counts[class_name] += 1

        # Update global ID counts
        for track in global_tracks:
            global_id = track['global_id']
            self.global_id_counts[global_id] += 1

        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            self.fps_history.append(fps)

            # Keep only recent FPS values
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)

    def record_detection(self, num_detections, detection_time=None):
        """
        Record detection metrics

        Args:
            num_detections: Number of objects detected
            detection_time: Time taken for detection (seconds)
        """
        self.detection_counts[self.frame_count] = num_detections

        if detection_time is not None:
            self.detection_times.append(detection_time)

    def record_tracking(self, num_tracks, tracking_time=None):
        """
        Record tracking metrics

        Args:
            num_tracks: Number of active tracks
            tracking_time: Time taken for tracking (seconds)
        """
        self.track_counts[self.frame_count] = num_tracks

        if tracking_time is not None:
            self.tracking_times.append(tracking_time)

    def record_reid(self, matched=False, reid_time=None):
        """
        Record ReID metrics

        Args:
            matched: Whether ReID matching was successful
            reid_time: Time taken for ReID (seconds)
        """
        self.reid_attempts += 1
        if matched:
            self.reid_matches += 1

        if reid_time is not None:
            self.reid_times.append(reid_time)

    def record_cross_camera_transfer(self):
        """Record a successful cross-camera object transfer"""
        self.cross_camera_transfers += 1

    def get_summary(self):
        """
        Get comprehensive metrics summary

        Returns:
            dict: metrics summary
        """
        elapsed_time = time.time() - self.start_time

        summary = {
            'runtime_seconds': elapsed_time,
            'total_frames': self.frame_count,
            'average_fps': np.mean(self.fps_history) if self.fps_history else 0,

            'detection': {
                'total_detections': sum(self.detection_counts.values()),
                'avg_detections_per_frame': np.mean(list(self.detection_counts.values())) if self.detection_counts else 0,
                'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0
            },

            'tracking': {
                'total_tracks': sum(self.track_counts.values()),
                'avg_tracks_per_frame': np.mean(list(self.track_counts.values())) if self.track_counts else 0,
                'avg_tracking_time': np.mean(self.tracking_times) if self.tracking_times else 0,
                'unique_global_ids': len(self.global_id_counts)
            },

            'reid': {
                'total_attempts': self.reid_attempts,
                'successful_matches': self.reid_matches,
                'match_rate': self.reid_matches / max(self.reid_attempts, 1),
                'avg_reid_time': np.mean(self.reid_times) if self.reid_times else 0
            },

            'cross_camera': {
                'transfers': self.cross_camera_transfers
            }
        }

        return summary

    def get_real_time_stats(self):
        """
        Get real-time statistics for monitoring

        Returns:
            dict: current statistics
        """
        current_fps = self.fps_history[-1] if self.fps_history else 0

        return {
            'current_fps': current_fps,
            'total_frames': self.frame_count,
            'active_global_ids': len([gid for gid, count in self.global_id_counts.items() if count > 0]),
            'reid_match_rate': self.reid_matches / max(self.reid_attempts, 1),
            'cross_camera_transfers': self.cross_camera_transfers
        }

    def export_metrics(self, filename):
        """
        Export metrics to file

        Args:
            filename: output filename
        """
        try:
            import json
            summary = self.get_summary()

            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Metrics exported to {filename}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def print_summary(self):
        """Print metrics summary to console"""
        summary = self.get_summary()

        print("\n" + "="*50)
        print("TRACKING SYSTEM METRICS SUMMARY")
        print("="*50)

        print(f"Runtime: {summary['runtime_seconds']:.2f}s")
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Average FPS: {summary['average_fps']:.2f}")

        print("\nDETECTION:")
        print(f"  Total Detections: {summary['detection']['total_detections']}")
        print(f"  Avg Detections/Frame: {summary['detection']['avg_detections_per_frame']:.2f}")
        print(f"  Avg Detection Time: {summary['detection']['avg_detection_time']:.4f}s")

        print("\nTRACKING:")
        print(f"  Total Tracks: {summary['tracking']['total_tracks']}")
        print(f"  Avg Tracks/Frame: {summary['tracking']['avg_tracks_per_frame']:.2f}")
        print(f"  Avg Tracking Time: {summary['tracking']['avg_tracking_time']:.4f}s")
        print(f"  Unique Global IDs: {summary['tracking']['unique_global_ids']}")

        print("\nRE-ID:")
        print(f"  Total Attempts: {summary['reid']['total_attempts']}")
        print(f"  Successful Matches: {summary['reid']['successful_matches']}")
        print(f"  Match Rate: {summary['reid']['match_rate']:.1%}")
        print(f"  Avg ReID Time: {summary['reid']['avg_reid_time']:.4f}s")

        print("\nCROSS-CAMERA:")
        print(f"  Transfers: {summary['cross_camera']['transfers']}")

        print("="*50)