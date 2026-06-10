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
