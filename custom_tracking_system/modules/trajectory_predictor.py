"""
Trajectory Prediction Module

Predicts future positions using a per-object Kalman Filter with a
constant-acceleration motion model (state = [x, y, vx, vy, ax, ay]).

Why Kalman over the previous exp-weighted-velocity approach:
  - Handles turns / acceleration / braking instead of assuming a fixed
    direction and speed.
  - Naturally smooths noisy detection/tracking jitter (measurement update).
  - dt-aware: works correctly with the irregular timestamps produced by
    real cameras, not just a fixed frame rate.

Public interface is unchanged so callers (AlertSystem, IncidentDetector,
ai_processor.py) do not need to change:
    update_trajectory(global_id, [x, y], unix_timestamp)
    predict(global_id) -> {'t+0.5s': [x, y], ..., 't+3.0s': [x, y]} | None
    predict_list(global_id) -> [[x, y], ...] | None
    get_velocity(global_id) -> [vx, vy] (px/s) | None
    get_speed(global_id) -> float (px/s) | None
"""

from __future__ import annotations

import time as _time
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

# Fixed prediction horizons (seconds into the future)
PRED_HORIZONS = (0.5, 1.0, 1.5, 2.0, 3.0)


class KalmanFilter2D:
    """Constant-acceleration Kalman Filter for a 2D point.

    State vector: [x, y, vx, vy, ax, ay]
    Measurement:  [x, y]  (bounding-box center, in pixels)
    """

    def __init__(self, position, process_noise_std: float = 50.0,
                 measurement_noise_std: float = 5.0):
        self.process_noise_std = process_noise_std

        self.x = np.zeros(6)
        self.x[0] = position[0]
        self.x[1] = position[1]

        # Large initial uncertainty on velocity/acceleration (unknown at start)
        self.P = np.diag([10.0, 10.0, 1e3, 1e3, 1e3, 1e3])

        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.R = np.eye(2) * (measurement_noise_std ** 2)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete white-noise-acceleration process noise (per axis)."""
        if dt <= 0:
            return np.zeros((6, 6))
        q = self.process_noise_std ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        # Single-axis [pos, vel, acc] noise block
        block = q * np.array([
            [dt4 / 4.0, dt3 / 2.0, dt2 / 2.0],
            [dt3 / 2.0, dt2,       dt],
            [dt2 / 2.0, dt,        1.0],
        ])
        Q = np.zeros((6, 6))
        for i, ax_i in enumerate((0, 2, 4)):  # x, vx, ax
            for j, ax_j in enumerate((0, 2, 4)):
                Q[ax_i, ax_j] = block[i, j]
        for i, ax_i in enumerate((1, 3, 5)):  # y, vy, ay
            for j, ax_j in enumerate((1, 3, 5)):
                Q[ax_i, ax_j] = block[i, j]
        return Q

    def predict(self, dt: float):
        """Advance the internal state by dt seconds (mutates state)."""
        if dt <= 0:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, measurement):
        """Incorporate a new [x, y] measurement (mutates state)."""
        z = np.asarray(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def project(self, dt: float) -> np.ndarray:
        """Return the projected [x, y] position `dt` seconds ahead,
        WITHOUT mutating the filter state."""
        F = self._F(dt)
        return (F @ self.x)[:2]

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4]

    @property
    def acceleration(self) -> np.ndarray:
        return self.x[4:6]


class TrajectoryPredictor:
    """
    Predicts object trajectories using a constant-acceleration Kalman Filter
    per global_id.

    update_trajectory() feeds new (position, unix_timestamp) observations
    into the filter (predict + update step, dt-aware).

    predict() returns a dict keyed by horizon label:
        {'t+0.5s': [x, y], 't+1.0s': [x, y], ..., 't+3.0s': [x, y]}

    predict_list() is a convenience wrapper returning the same data
    as a plain list — used by AlertSystem which expects list[list[x,y]].
    """

    def __init__(self, window_size: int = 10, pred_steps: int = None,
                 process_noise_std: float = 50.0,
                 measurement_noise_std: float = 5.0):
        """
        Args:
            window_size: Max observation history kept per object (used for
                          get_trajectory_history()/get_statistics(), not by
                          the Kalman Filter itself which is recursive).
            pred_steps:  Ignored — kept so existing call sites don't break.
            process_noise_std: How much we expect acceleration to change
                          between updates (px/s^2). Higher = filter trusts
                          new measurements more / reacts faster to turns.
            measurement_noise_std: Expected detector/tracker jitter (px).
        """
        self.window_size = window_size
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        # {global_id: {'positions': deque([x,y],...), 'timestamps': deque(float,...)}}
        self.trajectories: dict = {}
        # {global_id: KalmanFilter2D}
        self.filters: dict = {}
        # {global_id: float} — timestamp of the last update fed to the filter
        self._last_ts: dict = {}

        logger.info("TrajectoryPredictor initialised (Kalman constant-acceleration): "
                     "window=%d, horizons=%s s, process_noise_std=%.1f, measurement_noise_std=%.1f",
                     window_size, list(PRED_HORIZONS), process_noise_std, measurement_noise_std)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_trajectory(self, global_id: int, position: list,
                          timestamp: float = None):
        """
        Record a new position observation and update the Kalman Filter.

        Args:
            global_id:  Global object ID.
            position:   [x, y] pixel coordinates (bounding-box center).
            timestamp:  Unix time (float seconds). Defaults to time.time().
                        Pass time.time() from the caller for consistency
                        when batching multiple cameras.
        """
        ts = timestamp if timestamp is not None else _time.time()

        if global_id not in self.trajectories:
            self.trajectories[global_id] = {
                'positions':  deque(maxlen=self.window_size),
                'timestamps': deque(maxlen=self.window_size),
            }
        self.trajectories[global_id]['positions'].append(position)
        self.trajectories[global_id]['timestamps'].append(ts)

        kf = self.filters.get(global_id)
        if kf is None:
            self.filters[global_id] = KalmanFilter2D(
                position, self.process_noise_std, self.measurement_noise_std)
            self._last_ts[global_id] = ts
            return

        dt = ts - self._last_ts.get(global_id, ts)
        if dt > 1e-6:
            kf.predict(dt)
            self._last_ts[global_id] = ts
        kf.update(position)

    def predict(self, global_id: int) -> dict | None:
        """
        Predict future positions at PRED_HORIZONS time horizons.

        Returns:
            {'t+0.5s': [x, y], 't+1.0s': [x, y], ..., 't+3.0s': [x, y]}
            None if fewer than 2 observations are stored.
        """
        traj = self.trajectories.get(global_id)
        if not traj or len(traj['positions']) < 2:
            return None
        kf = self.filters.get(global_id)
        if kf is None:
            return None
        return {f't+{h}s': kf.project(h).tolist() for h in PRED_HORIZONS}

    def predict_list(self, global_id: int) -> list | None:
        """
        Predicted positions as a plain list of [x, y] points.
        Preserves the interface expected by AlertSystem.check_alerts().
        """
        pred = self.predict(global_id)
        return list(pred.values()) if pred else None

    def get_velocity(self, global_id: int):
        """Return current Kalman velocity estimate [vx, vy] in px/s, or None."""
        kf = self.filters.get(global_id)
        if kf is None:
            return None
        return kf.velocity

    def get_acceleration(self, global_id: int):
        """Return current Kalman acceleration estimate [ax, ay] in px/s^2, or None."""
        kf = self.filters.get(global_id)
        if kf is None:
            return None
        return kf.acceleration

    def get_speed(self, global_id: int):
        """Return scalar speed in px/s, or None."""
        vel = self.get_velocity(global_id)
        return float(np.linalg.norm(vel)) if vel is not None else None

    def get_trajectory_history(self, global_id: int) -> dict:
        traj = self.trajectories.get(global_id)
        if traj is None:
            return {'positions': [], 'timestamps': []}
        return {
            'positions':  list(traj['positions']),
            'timestamps': list(traj['timestamps']),
        }

    def clear_trajectory(self, global_id: int):
        self.trajectories.pop(global_id, None)
        self.filters.pop(global_id, None)
        self._last_ts.pop(global_id, None)

    def get_all_trajectories(self) -> dict:
        return self.trajectories.copy()

    def get_statistics(self) -> dict:
        total = len(self.trajectories)
        avg_len = 0.0
        if total:
            avg_len = float(np.mean([len(t['positions'])
                                     for t in self.trajectories.values()]))
        return {
            'total_objects':         total,
            'avg_trajectory_length': avg_len,
            'window_size':           self.window_size,
            'pred_horizons_s':       list(PRED_HORIZONS),
            'model':                 'kalman_constant_acceleration',
        }


class LearnedTrajectoryPredictor:
    """Seq2Seq GRU trajectory predictor (world meters), trained offline on
    CARLA data — see train_trajectory_predictor.py.

    Operates on world-frame [x, y, vx, vy] (meters / m/s), not pixels.
    If the checkpoint file does not exist (or torch/model import fails),
    `self.available` is False and predict() always returns None — callers
    must fall back to Kalman-only behaviour.
    """

    def __init__(self, checkpoint_path: str = 'weights/trajectory_gru.pth'):
        self.available = False
        self.checkpoint_path = checkpoint_path
        self.histories: dict = {}  # {global_id: deque(maxlen=T_OBS) of step-dicts}

        try:
            import os
            import torch
            from .trajectory_gru import (
                TrajectoryGRU, build_feature_vector, INTENT_LABELS, T_OBS,
            )

            if not os.path.exists(checkpoint_path):
                logger.info("LearnedTrajectoryPredictor: checkpoint not found at %s "
                             "-> GRU predictions disabled (Kalman-only)", checkpoint_path)
                return

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            self._torch = torch
            self._build_feature_vector = build_feature_vector
            self._intent_labels = INTENT_LABELS
            self.t_obs = checkpoint.get('t_obs', T_OBS)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model = TrajectoryGRU(hidden_size=checkpoint.get('hidden_size', 96),
                                        num_modes=checkpoint.get('num_modes', 3))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.norm_mean = np.asarray(checkpoint['norm_mean'], dtype=np.float32)
            self.norm_std = np.asarray(checkpoint['norm_std'], dtype=np.float32)
            self.pred_horizons = checkpoint.get('pred_horizons', PRED_HORIZONS)
            self.class_list = checkpoint.get('class_list', ('car', 'motorcycle', 'person'))

            self.available = True
            logger.info("LearnedTrajectoryPredictor: loaded checkpoint %s (t_obs=%d)",
                         checkpoint_path, self.t_obs)
        except Exception:
            logger.exception("LearnedTrajectoryPredictor: failed to load checkpoint %s "
                              "-> GRU predictions disabled (Kalman-only)", checkpoint_path)
            self.available = False

    def update(self, global_id: int, world_pos, world_vel, obj_class: str,
               neighbors: list, timestamp: float = None):
        """Append a new world-frame observation for `global_id`.

        Args:
            world_pos: (x, y) meters.
            world_vel: (vx, vy) m/s.
            obj_class: 'car' | 'motorcycle' | 'person' (or unknown).
            neighbors: list of up to NUM_NEIGHBORS (dx, dy, dvx, dvy) tuples.
        """
        if not self.available:
            return
        if global_id not in self.histories:
            self.histories[global_id] = deque(maxlen=self.t_obs)
        self.histories[global_id].append({
            'pos': world_pos,
            'vel': world_vel,
            'cls': obj_class,
            'neighbors': neighbors,
        })

    def predict(self, global_id: int) -> dict | None:
        """
        Returns:
            {'modes': [[[x,y] x len(pred_horizons)] x num_modes],  # world meters
             'probs': [float x num_modes],
             'intent': str,
             'last_pos': [x, y]}
            or None if unavailable / not enough history.
        """
        if not self.available:
            return None
        hist = self.histories.get(global_id)
        if hist is None or len(hist) < self.t_obs:
            return None

        torch = self._torch
        obs_seq = np.stack([
            self._build_feature_vector(s['pos'], s['vel'], s['cls'], s['neighbors'],
                                        self.norm_mean, self.norm_std)
            for s in hist
        ])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)  # (1, T_OBS, input_dim)
            pred_deltas, mode_logits, intent_logits = self.model(obs_tensor)
            probs = torch.softmax(mode_logits, dim=-1)[0].cpu().numpy()
            intent_idx = int(torch.argmax(intent_logits, dim=-1)[0].item())
            deltas = pred_deltas[0].cpu().numpy()  # (num_modes, num_horizons, 2)

        last_pos = np.array(hist[-1]['pos'], dtype=np.float32)
        pos_std = self.norm_std[:2]
        modes = (deltas * pos_std + last_pos).tolist()  # de-normalize -> world meters

        return {
            'modes': modes,
            'probs': probs.tolist(),
            'intent': self._intent_labels[intent_idx],
            'last_pos': last_pos.tolist(),
        }

    def clear_trajectory(self, global_id: int):
        self.histories.pop(global_id, None)


class EnsembleTrajectoryPredictor:
    """Drop-in replacement for TrajectoryPredictor that blends the Kalman
    constant-acceleration filter (pixel-space, always available) with a
    learned Seq2Seq GRU model (world-space, optional).

    Public interface (predict/predict_list/get_velocity/...) is unchanged
    and remains pixel-space, so AlertSystem/IncidentDetector/main.py do not
    need to change. If no calibration is supplied for a camera, or the GRU
    checkpoint is unavailable, behaviour is identical to plain
    TrajectoryPredictor (Kalman-only).
    """

    def __init__(self, window_size: int = 10, calibration: dict | None = None,
                 gru_checkpoint: str = 'weights/trajectory_gru.pth'):
        self.kalman = TrajectoryPredictor(window_size=window_size)
        self.learned = LearnedTrajectoryPredictor(gru_checkpoint)
        self.calibration = calibration or {}

        # {global_id: (world_pos, timestamp)} — for finite-difference world velocity
        self._world_history: dict = {}
        # {global_id: camera_id} — needed to convert GRU world predictions back to pixels
        self._camera_of: dict = {}

        if self.calibration and self.learned.available:
            logger.info("EnsembleTrajectoryPredictor: Kalman + GRU ensemble active "
                         "(%d camera calibration(s))", len(self.calibration))
        else:
            logger.info("EnsembleTrajectoryPredictor: Kalman-only "
                         "(calibration=%s, gru_available=%s)",
                         bool(self.calibration), self.learned.available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_trajectory(self, global_id: int, position: list, timestamp: float = None,
                          camera_id: str = None, obj_class: str = 'car',
                          all_tracks: list | None = None):
        """
        Args:
            position:   [x, y] pixel coordinates (bounding-box center).
            camera_id:  CAM_001/etc — required (with calibration) to feed the GRU.
            obj_class:  'car' | 'motorcycle' | 'person'.
            all_tracks: optional list of global_track dicts (this frame, all
                        cameras) — used to build neighbor features for the GRU.
        """
        ts = timestamp if timestamp is not None else _time.time()

        self.kalman.update_trajectory(global_id, position, ts)

        if camera_id is not None and self.learned.available:
            calib = self.calibration.get(camera_id)
            if calib is not None:
                self._camera_of[global_id] = camera_id
                self._update_learned(global_id, position, ts, camera_id, obj_class,
                                      all_tracks, calib)

    def _update_learned(self, global_id, position, ts, camera_id, obj_class, all_tracks, calib):
        world_pos = calib.pixel_to_world(position[0], position[1])

        prev = self._world_history.get(global_id)
        if prev is not None:
            prev_pos, prev_ts = prev
            dt = ts - prev_ts
            if dt > 1e-6:
                world_vel = ((world_pos[0] - prev_pos[0]) / dt,
                              (world_pos[1] - prev_pos[1]) / dt)
            else:
                world_vel = (0.0, 0.0)
        else:
            world_vel = (0.0, 0.0)
        self._world_history[global_id] = (world_pos, ts)

        neighbors = self._compute_neighbors(global_id, world_pos, camera_id, all_tracks, calib)
        self.learned.update(global_id, world_pos, world_vel, obj_class, neighbors, ts)

    def _compute_neighbors(self, global_id, world_pos, camera_id, all_tracks, calib,
                           num_neighbors: int = 3):
        if not all_tracks:
            return []

        candidates = []
        for track in all_tracks:
            if track.get('camera_id') != camera_id:
                continue
            if track.get('global_id') == global_id:
                continue
            box = track.get('box')
            if box is None:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            other_world = calib.pixel_to_world(cx, cy)
            dx, dy = other_world[0] - world_pos[0], other_world[1] - world_pos[1]
            # Neighbor velocity not tracked here -> approximate as 0 (relative
            # velocity then equals -ego_vel, a reasonable static-neighbor prior).
            candidates.append((dx * dx + dy * dy, (dx, dy, 0.0, 0.0)))

        candidates.sort(key=lambda c: c[0])
        return [rel for _, rel in candidates[:num_neighbors]]

    def predict(self, global_id: int) -> dict | None:
        kalman_pred = self.kalman.predict(global_id)
        if kalman_pred is None:
            return None

        if not self.learned.available:
            return kalman_pred

        learned_pred = self.learned.predict(global_id)
        if learned_pred is None:
            return kalman_pred

        camera_id = self._camera_of.get(global_id)
        calib = self.calibration.get(camera_id) if camera_id else None
        if calib is None:
            return kalman_pred

        probs = learned_pred['probs']
        best_mode = int(np.argmax(probs))
        prob = probs[best_mode]
        best_path = learned_pred['modes'][best_mode]

        blended = {}
        for i, h in enumerate(PRED_HORIZONS):
            key = f't+{h}s'
            kalman_xy = np.asarray(kalman_pred[key], dtype=float)
            if i < len(best_path):
                gru_pixel = calib.world_to_pixel(best_path[i][0], best_path[i][1])
            else:
                gru_pixel = None

            if gru_pixel is None:
                blended[key] = kalman_xy.tolist()
            else:
                gru_xy = np.asarray(gru_pixel, dtype=float)
                blended[key] = (prob * gru_xy + (1.0 - prob) * kalman_xy).tolist()

        return blended

    def predict_list(self, global_id: int) -> list | None:
        pred = self.predict(global_id)
        return list(pred.values()) if pred else None

    def predict_modes(self, global_id: int) -> dict | None:
        """Multi-modal GRU prediction in pixel-space, or None if unavailable.

        Returns:
            {'modes': [[[x,y] x len(pred_horizons)] x num_modes],  # pixels
             'probs': [float x num_modes],
             'intent': str}
        """
        if not self.learned.available:
            return None
        learned_pred = self.learned.predict(global_id)
        if learned_pred is None:
            return None

        camera_id = self._camera_of.get(global_id)
        calib = self.calibration.get(camera_id) if camera_id else None
        if calib is None:
            return None

        pixel_modes = []
        for path in learned_pred['modes']:
            pixel_path = []
            for x, y in path:
                px = calib.world_to_pixel(x, y)
                pixel_path.append(list(px) if px is not None else None)
            pixel_modes.append(pixel_path)

        return {
            'modes': pixel_modes,
            'probs': learned_pred['probs'],
            'intent': learned_pred['intent'],
        }

    def get_velocity(self, global_id: int):
        return self.kalman.get_velocity(global_id)

    def get_acceleration(self, global_id: int):
        return self.kalman.get_acceleration(global_id)

    def get_speed(self, global_id: int):
        return self.kalman.get_speed(global_id)

    def get_trajectory_history(self, global_id: int) -> dict:
        return self.kalman.get_trajectory_history(global_id)

    def clear_trajectory(self, global_id: int):
        self.kalman.clear_trajectory(global_id)
        self.learned.clear_trajectory(global_id)
        self._world_history.pop(global_id, None)
        self._camera_of.pop(global_id, None)

    def get_all_trajectories(self) -> dict:
        return self.kalman.get_all_trajectories()

    def get_statistics(self) -> dict:
        stats = self.kalman.get_statistics()
        stats['model'] = 'kalman_ca + gru_ensemble' if self.learned.available else 'kalman_ca'
        stats['gru_available'] = self.learned.available
        stats['calibrated_cameras'] = list(self.calibration.keys())
        return stats
