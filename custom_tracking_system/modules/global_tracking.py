"""
Global Tracking Module
Manages cross-camera object tracking with global IDs
"""

import numpy as np
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GlobalTracker:
    """
    Manages global object IDs across multiple cameras
    """

    def __init__(self, reid_extractor, match_threshold=0.5, max_age=300):
        """
        Initialize global tracker

        Args:
            reid_extractor: ReIDExtractor instance
            match_threshold: Threshold for ReID matching
            max_age: Maximum age for global tracks (seconds)
        """
        self.reid_extractor = reid_extractor
        self.match_threshold = match_threshold
        self.max_age = max_age

        # Global tracks: {global_id: track_info}
        self.global_tracks = {}

        # Next available global ID
        self.next_global_id = 1000

        logger.info(f"GlobalTracker initialized with threshold={match_threshold}")

    def process_camera_tracks(self, camera_id, frame, local_tracks):
        """
        Process local tracks from a camera and assign global IDs

        Args:
            camera_id: Camera identifier
            frame: numpy array - camera frame
            local_tracks: list of local track dicts

        Returns:
            list: global tracks with assigned IDs
        """
        global_tracks_output = []

        for local_track in local_tracks:
            box = local_track['box']

            # Extract ReID feature
            feature = self.reid_extractor.extract_feature(frame, box)
            if feature is None:
                continue

            # Try to match with existing global tracks
            matched_global_id = self.reid_extractor.match_with_gallery(
                feature, threshold=self.match_threshold)

            if matched_global_id is not None:
                global_id = matched_global_id
            else:
                # Create new global ID
                global_id = self.next_global_id
                self.next_global_id += 1

            # Update gallery
            self.reid_extractor.add_to_gallery(global_id, feature)

            # Update global track
            self._update_global_track(global_id, camera_id, local_track)

            global_tracks_output.append({
                'global_id': global_id,
                'camera_id': camera_id,
                'box': box,
                'class': local_track['class'],
                'local_track_id': local_track['track_id']
            })

        return global_tracks_output

    def _update_global_track(self, global_id, camera_id, local_track):
        """
        Update information for a global track

        Args:
            global_id: Global object ID
            camera_id: Camera ID
            local_track: Local track information
        """
        now = datetime.now()

        if global_id not in self.global_tracks:
            # Create new global track
            self.global_tracks[global_id] = {
                'global_id': global_id,
                'class': local_track['class'],
                'first_seen': now,
                'last_seen': now,
                'camera_history': defaultdict(list),
                'total_cameras': set()
            }

        track = self.global_tracks[global_id]
        track['last_seen'] = now
        track['total_cameras'].add(camera_id)

        # Add to camera history
        track['camera_history'][camera_id].append({
            'timestamp': now,
            'box': local_track['box'],
            'local_track_id': local_track['track_id']
        })

        # Keep only recent history (last 100 entries per camera)
        if len(track['camera_history'][camera_id]) > 100:
            track['camera_history'][camera_id].pop(0)

    def get_active_tracks(self, timeout_seconds=30):
        """
        Get currently active global tracks

        Args:
            timeout_seconds: Maximum age for active tracks

        Returns:
            dict: active global tracks
        """
        now = datetime.now()
        active_tracks = {}

        for global_id, track in self.global_tracks.items():
            elapsed = (now - track['last_seen']).total_seconds()

            if elapsed < timeout_seconds:
                active_tracks[global_id] = track

        return active_tracks

    def get_track_trajectory(self, global_id, camera_id=None):
        """
        Get trajectory of a global track

        Args:
            global_id: Global object ID
            camera_id: Specific camera (None for all cameras)

        Returns:
            list: trajectory points [(timestamp, position), ...]
        """
        if global_id not in self.global_tracks:
            return []

        track = self.global_tracks[global_id]
        trajectory = []

        cameras = [camera_id] if camera_id else track['camera_history'].keys()

        for cam_id in cameras:
            if cam_id in track['camera_history']:
                for entry in track['camera_history'][cam_id]:
                    # Calculate center position from box
                    box = entry['box']
                    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

                    trajectory.append({
                        'timestamp': entry['timestamp'],
                        'position': center,
                        'camera_id': cam_id
                    })

        # Sort by timestamp
        trajectory.sort(key=lambda x: x['timestamp'])

        return trajectory

    def get_track_info(self, global_id):
        """
        Get detailed information about a global track

        Args:
            global_id: Global object ID

        Returns:
            dict: track information or None
        """
        return self.global_tracks.get(global_id)

    def get_all_tracks(self):
        """
        Get all global tracks

        Returns:
            dict: all tracks
        """
        return self.global_tracks.copy()

    def get_statistics(self):
        """
        Get tracking statistics

        Returns:
            dict: statistics
        """
        active_tracks = self.get_active_tracks()

        return {
            'total_global_ids': len(self.global_tracks),
            'active_tracks': len(active_tracks),
            'total_cameras_used': len(set(
                cam_id for track in self.global_tracks.values()
                for cam_id in track['total_cameras']
            )),
            'gallery_size': self.reid_extractor.get_gallery_size()
        }

    def reset(self):
        """Reset global tracker state"""
        self.global_tracks.clear()
        self.next_global_id = 1000
        self.reid_extractor.clear_gallery()
        logger.info("Global tracker reset")