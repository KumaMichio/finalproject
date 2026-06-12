"""
Multi-Object Tracking Module

ByteTrackWrapper  — production tracker (boxmot.ByteTrack + Kalman + Hungarian)
SimpleTracker     — legacy IoU-greedy fallback, kept for reference
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Default class map (COCO indices) — only used if no class_map is supplied.
# Pass detector.CLASSES explicitly when the detector uses a different scheme
# (e.g. the VN-finetuned checkpoint's CLASSES_VN = {0:person,1:car,2:motorcycle,3:bus,4:truck}).
_CLS_MAP = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class ByteTrackWrapper:
    """
    Per-camera tracker wrapping boxmot.ByteTrack.
    Drop-in replacement for SimpleTracker — update() takes an extra `frame` arg.

    ByteTrack advantages over IoU-greedy:
      - Kalman Filter predicts position under occlusion
      - Hungarian matching (optimal, not greedy)
      - Dual-threshold: keeps low-confidence detections as tentative tracks
        to survive brief occlusions without ID swap
    """

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30,
                 class_map: dict | None = None):
        self._init_kwargs = dict(
            track_thresh=track_activation_threshold,
            track_buffer=lost_track_buffer,
            match_thresh=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        # Must match the class-index scheme of the ObjectDetector feeding this
        # tracker (detector.CLASSES) — boxmot only returns the numeric class_id,
        # so the wrong map silently relabels classes (e.g. VN cls_id=1 'car'
        # would show up as 'unknown' under the COCO map).
        self.class_map = class_map if class_map is not None else _CLS_MAP
        self._history: dict = {}  # {track_id: {positions, timestamps, speeds, class}}
        self._init_tracker()
        logger.info("ByteTrackWrapper initialised (frame_rate=%d, buffer=%d)",
                    frame_rate, lost_track_buffer)

    def _init_tracker(self):
        from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
        self.tracker = ByteTrack(**self._init_kwargs)

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracker with new detections.

        Args:
            detections: list of {'box': [x1,y1,x2,y2], 'confidence': float,
                                  'class': str, 'class_id': int}
            frame: current camera frame (H×W×3 BGR numpy array)

        Returns:
            list of {'track_id': int, 'box': [x1,y1,x2,y2], 'class': str,
                     'positions': list, 'timestamps': list, 'speeds': list}
        """
        if detections:
            dets_arr = np.array(
                [[*d['box'], d['confidence'], d.get('class_id', 0)]
                 for d in detections],
                dtype=np.float32,
            )
        else:
            dets_arr = np.empty((0, 6), dtype=np.float32)

        raw = self.tracker.update(dets_arr, frame)

        now = datetime.now()
        output = []

        if raw is None or len(raw) == 0:
            return output

        for row in raw:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            track_id = int(row[4])
            cls_id   = int(row[6])

            box      = [x1, y1, x2, y2]
            cls_name = self.class_map.get(cls_id, 'unknown')
            center   = [(x1 + x2) / 2, (y1 + y2) / 2]

            if track_id not in self._history:
                self._history[track_id] = {
                    'class': cls_name,
                    'positions': [],
                    'timestamps': [],
                    'speeds': [],
                }
            hist = self._history[track_id]
            hist['positions'].append(center)
            hist['timestamps'].append(now)
            if len(hist['positions']) > 100:
                hist['positions'].pop(0)
                hist['timestamps'].pop(0)

            if len(hist['positions']) >= 2:
                dt = (hist['timestamps'][-1] - hist['timestamps'][-2]).total_seconds()
                if dt > 1e-6:
                    d = np.linalg.norm(
                        np.array(hist['positions'][-1]) - np.array(hist['positions'][-2])
                    )
                    hist['speeds'].append(d / dt)
                    if len(hist['speeds']) > 30:
                        hist['speeds'].pop(0)

            output.append({
                'track_id':  track_id,
                'box':       box,
                'class':     cls_name,
                'positions': list(hist['positions']),
                'timestamps': list(hist['timestamps']),
                'speeds':    list(hist['speeds']),
            })

        logger.debug("ByteTrackWrapper: %d active tracks", len(output))
        return output

    def reset(self):
        """Reset tracker state (re-initialises the underlying ByteTrack)."""
        self._history.clear()
        self._init_tracker()
        logger.info("ByteTrackWrapper reset")

    def get_all_tracks(self) -> dict:
        return {tid: dict(h) for tid, h in self._history.items()}

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