"""
Multi-Object Tracking Module
Implements simple tracking algorithm for single camera
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleTracker:
    """
    Simple multi-object tracker using IoU-based matching
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = {}  # {track_id: track_info}
        self.next_id = 0
        self.frame_count = 0

        logger.info(f"SimpleTracker initialized: max_age={max_age}, min_hits={min_hits}")

    def update(self, detections):
        """
        Update tracks with new detections

        Args:
            detections: list of {'box': [x1,y1,x2,y2], 'confidence': float, 'class': str}

        Returns:
            list: active tracks [{'track_id': int, 'box': [...], 'class': str, 'positions': [...]}]
        """
        self.frame_count += 1

        # Get active tracks (not too old)
        active_tracks = [t for t in self.tracks.values() if t['age'] < self.max_age]

        matched_detections = set()
        matched_tracks = set()

        # Match detections to existing tracks
        if len(active_tracks) > 0 and len(detections) > 0:
            iou_matrix = self._compute_iou_matrix(active_tracks, detections)

            for track_idx, det_idx in self._greedy_matching(iou_matrix):
                track = active_tracks[track_idx]
                det = detections[det_idx]

                # Update track
                track['box'] = det['box']
                track['age'] = 0
                track['hits'] += 1
                now = datetime.now()
                new_pos = self._box_center(det['box'])
                track['positions'].append(new_pos)
                track['timestamps'].append(now)

                # Tính speed (pixels/second)
                if len(track['positions']) >= 2:
                    dt = (track['timestamps'][-1] - track['timestamps'][-2]).total_seconds()
                    if dt > 1e-6:
                        d = np.linalg.norm(
                            np.array(track['positions'][-1]) - np.array(track['positions'][-2])
                        )
                        track['speeds'].append(d / dt)
                        if len(track['speeds']) > 30:
                            track['speeds'].pop(0)

                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)

        # Age unmatched tracks
        for idx, track in enumerate(active_tracks):
            if idx not in matched_tracks:
                track['age'] += 1

        # Create new tracks from unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'box': det['box'],
                    'class': det['class'],
                    'age': 0,
                    'hits': 1,
                    'positions': [self._box_center(det['box'])],
                    'timestamps': [datetime.now()],
                    'speeds': [],
                }
                self.next_id += 1

        # Return confirmed tracks
        output = []
        for track in self.tracks.values():
            if track['hits'] >= self.min_hits and track['age'] == 0:
                output.append({
                    'track_id': track['track_id'] if 'track_id' in track else track['id'],
                    'box': track['box'],
                    'class': track['class'],
                    'positions': track['positions'],
                    'timestamps': track.get('timestamps', []),
                    'speeds': track.get('speeds', []),
                })

        logger.debug(f"Updated {len(output)} active tracks")
        return output

    def _compute_iou_matrix(self, tracks, detections):
        """
        Compute IoU matrix between tracks and detections

        Args:
            tracks: list of track dicts
            detections: list of detection dicts

        Returns:
            numpy array: IoU matrix (n_tracks, n_detections)
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track['box'], det['box'])

        return iou_matrix

    def _iou(self, box1, box2):
        """
        Calculate Intersection over Union

        Args:
            box1, box2: [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def _greedy_matching(self, iou_matrix):
        """
        Greedy matching algorithm

        Args:
            iou_matrix: numpy array (n_tracks, n_detections)

        Returns:
            list: [(track_idx, det_idx), ...]
        """
        matches = []
        used_dets = set()

        for i in range(iou_matrix.shape[0]):
            best_j = np.argmax(iou_matrix[i, :])
            if iou_matrix[i, best_j] > self.iou_threshold and best_j not in used_dets:
                matches.append((i, best_j))
                used_dets.add(best_j)

        return matches

    def _box_center(self, box):
        """
        Calculate center of bounding box

        Args:
            box: [x1, y1, x2, y2]

        Returns:
            list: [x, y]
        """
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def get_track_info(self, track_id):
        """
        Get information about a specific track

        Args:
            track_id: Track ID

        Returns:
            dict: Track information or None
        """
        return self.tracks.get(track_id)

    def get_all_tracks(self):
        """
        Get all current tracks

        Returns:
            dict: All tracks
        """
        return self.tracks.copy()

    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        logger.info("Tracker reset")