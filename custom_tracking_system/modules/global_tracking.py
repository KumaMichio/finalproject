"""
Global Tracking Module
Manages cross-camera object tracking with global IDs.
Includes Spatio-Temporal Filter to reject physically impossible cross-camera matches.
"""

import time as _time
import numpy as np
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GlobalTracker:
    """
    Manages global object IDs across multiple cameras
    """

    def __init__(self, reid_extractor, match_threshold=0.5, max_age=300,
                 camera_topology: dict = None):
        """
        Initialize global tracker.

        Args:
            reid_extractor:   ReIDExtractor instance
            match_threshold:  Cosine-similarity threshold for ReID matching
            max_age:          Max seconds to keep an inactive global track
            camera_topology:  Optional dict describing feasible travel times between cameras.
                              Format: {(cam_a, cam_b): (min_seconds, max_seconds)}
                              Both directions must be listed separately.
                              Example:
                                {('CAM_001', 'CAM_002'): (5, 60),
                                 ('CAM_002', 'CAM_001'): (5, 60)}
                              If None or empty, no spatio-temporal filtering is applied.
        """
        self.reid_extractor = reid_extractor
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.camera_topology: dict = camera_topology or {}

        # Global tracks: {global_id: track_info}
        self.global_tracks = {}

        # Next available global ID
        self.next_global_id = 1000

        # Spatio-temporal state: last seen camera + unix timestamp per global_id
        # {global_id: {'camera_id': str, 'timestamp': float}}
        self._last_seen_info: dict = {}

        # ReID-throttle cache: {(camera_id, local_track_id): global_id} — lets
        # callers skip extract_feature() on most frames for already-identified
        # tracks (see process_camera_tracks(reid_track_ids=...)).
        self._track_to_global: dict = {}

        logger.info(
            "GlobalTracker initialised: threshold=%.2f, topology_pairs=%d",
            match_threshold, len(self.camera_topology)
        )

    def process_camera_tracks(self, camera_id, frame, local_tracks, reid_track_ids=None):
        """
        Process local tracks from a camera and assign global IDs

        Args:
            camera_id: Camera identifier
            frame: numpy array - camera frame
            local_tracks: list of local track dicts
            reid_track_ids: optional set of local_track_id values for which
                ReID extraction should run THIS frame. Tracks whose
                local_track_id is NOT in this set reuse the global_id they
                were assigned last time ReID ran for them (cache keyed by
                (camera_id, local_track_id)), skipping extract_feature() —
                used to throttle the per-object OSNet forward pass to every
                Nth frame. If None (default), ReID always runs for every
                track (original behaviour, used by main.py).

        Returns:
            list: global tracks with assigned IDs
        """
        global_tracks_output = []

        # Global IDs already assigned to a track from THIS camera in THIS
        # frame. Two simultaneous detections from the same camera can never
        # be the same physical object, no matter how similar their ReID
        # features look (low-resolution crops can score above the match
        # threshold against each other) — so re-matching to an ID already
        # used this frame is rejected and a fresh ID is created instead.
        used_global_ids_this_frame = set()

        for local_track in local_tracks:
            box = local_track['box']
            track_id = local_track['track_id']
            obj_class = local_track.get('class', 'unknown')
            cache_key = (camera_id, track_id)
            cached_global_id = self._track_to_global.get(cache_key)

            run_reid = reid_track_ids is None or track_id in reid_track_ids

            if not run_reid and cached_global_id is not None:
                # Reuse the previously-assigned global ID without running
                # ReID this frame.
                global_id = cached_global_id
                used_global_ids_this_frame.add(global_id)
                self._update_global_track(global_id, camera_id, local_track)
                global_tracks_output.append(
                    self._make_output(global_id, camera_id, box, local_track))
                continue

            # Extract ReID feature — pass object class so DualReIDExtractor
            # can choose person vs vehicle model automatically
            feature = self.reid_extractor.extract_feature(frame, box, obj_class)
            if feature is None:
                if cached_global_id is not None:
                    global_id = cached_global_id
                    used_global_ids_this_frame.add(global_id)
                    self._update_global_track(global_id, camera_id, local_track)
                    global_tracks_output.append(
                        self._make_output(global_id, camera_id, box, local_track))
                continue

            # Try to match with existing global tracks. Only compare against
            # gallery entries from the same embedding-space group (e.g. don't
            # match a car's vehicle-model feature against a pedestrian's
            # person-model feature — those cosine similarities are meaningless).
            group = self.reid_extractor.group_for_class(obj_class)
            matched_global_id = self.reid_extractor.match_with_gallery(
                feature, threshold=self.match_threshold, group=group)

            # Accept match only when spatio-temporally feasible and not
            # already claimed by another track from this camera this frame.
            if (matched_global_id is not None and
                    matched_global_id not in used_global_ids_this_frame and
                    self._is_feasible_match(matched_global_id, camera_id)):
                global_id = matched_global_id
            else:
                # Create new global ID (also covers infeasible cross-camera
                # match and same-frame collisions)
                global_id = self.next_global_id
                self.next_global_id += 1

            used_global_ids_this_frame.add(global_id)

            # Update gallery and track state
            self.reid_extractor.add_to_gallery(global_id, feature, group=group)
            self._update_global_track(global_id, camera_id, local_track)
            self._track_to_global[cache_key] = global_id

            global_tracks_output.append(
                self._make_output(global_id, camera_id, box, local_track))

        return global_tracks_output

    @staticmethod
    def _make_output(global_id, camera_id, box, local_track):
        return {
            'global_id':      global_id,
            'camera_id':      camera_id,
            'box':            box,
            'class':          local_track['class'],
            'local_track_id': local_track['track_id'],
            # Propagate motion data so IncidentDetector can use speeds
            'speeds':         local_track.get('speeds', []),
            'positions':      local_track.get('positions', []),
            'timestamps':     local_track.get('timestamps', []),
        }

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

        # Update spatio-temporal state (used by _is_feasible_match on next frame)
        self._last_seen_info[global_id] = {
            'camera_id': camera_id,
            'timestamp': _time.time(),
        }

        # Add to camera history
        track['camera_history'][camera_id].append({
            'timestamp': now,
            'box': local_track['box'],
            'local_track_id': local_track['track_id']
        })

        # Keep only recent history (last 100 entries per camera)
        if len(track['camera_history'][camera_id]) > 100:
            track['camera_history'][camera_id].pop(0)

    # ------------------------------------------------------------------
    # Spatio-Temporal Filter
    # ------------------------------------------------------------------

    def _is_feasible_match(self, global_id: int, target_camera: str) -> bool:
        """
        Return True if the elapsed time since this global_id was last seen
        is consistent with physically traveling to target_camera.

        Always returns True when:
          - The object has never been seen before (first appearance).
          - The last camera equals target_camera (same-camera re-match).
          - No topology entry exists for the camera pair (unconstrained).
        """
        info = self._last_seen_info.get(global_id)
        if info is None:
            return True  # first appearance — no spatial constraint yet

        last_cam = info['camera_id']
        if last_cam == target_camera:
            return True  # same camera, no topology check needed

        key = (last_cam, target_camera)
        if key not in self.camera_topology:
            return True  # no constraint defined for this pair

        elapsed = _time.time() - info['timestamp']
        min_t, max_t = self.camera_topology[key]
        feasible = min_t <= elapsed <= max_t
        if not feasible:
            logger.debug(
                "ST-filter rejected match gid=%d: %s→%s elapsed=%.1fs (allowed %.0f–%.0f s)",
                global_id, last_cam, target_camera, elapsed, min_t, max_t,
            )
        return feasible

    def is_feasible_transition(self, cam_a: str, t_a: float,
                               cam_b: str, t_b: float) -> bool:
        """
        Public utility: check whether an object could travel from cam_a at time t_a
        to cam_b at time t_b given the configured topology.
        t_a, t_b are Unix timestamps (float seconds).
        """
        key = (cam_a, cam_b)
        if key not in self.camera_topology:
            return True
        min_t, max_t = self.camera_topology[key]
        delta = abs(t_b - t_a)
        return min_t <= delta <= max_t

    # ------------------------------------------------------------------

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
        """Reset global tracker state."""
        self.global_tracks.clear()
        self._last_seen_info.clear()
        self._track_to_global.clear()
        self.next_global_id = 1000
        self.reid_extractor.clear_gallery()
        logger.info("Global tracker reset")