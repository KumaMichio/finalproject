"""
Trajectory Prediction Module

Predicts future positions using exponentially-weighted velocity (px/s).
Fixes vs. original linear version:
  - velocity uses all window_size points (not just last 2)
  - units are px/s tied to real timestamps, not px/frame
  - prediction horizons are real-time seconds (0.5–3 s), not frame counts
"""

import time as _time
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

# Fixed prediction horizons (seconds into the future)
PRED_HORIZONS = (0.5, 1.0, 1.5, 2.0, 3.0)


class TrajectoryPredictor:
    """
    Predicts object trajectories using exponentially-weighted velocity.

    update_trajectory() stores (position, unix_timestamp) pairs.
    predict() returns a dict keyed by horizon label:
        {'t+0.5s': [x, y], 't+1.0s': [x, y], ..., 't+3.0s': [x, y]}

    predict_list() is a convenience wrapper returning the same data
    as a plain list — used by AlertSystem which expects list[list[x,y]].
    """

    def __init__(self, window_size: int = 10, pred_steps: int = None):
        """
        Args:
            window_size: Max observation history per object.
            pred_steps:  Ignored — kept so existing call sites don't break.
        """
        self.window_size = window_size
        # {global_id: {'positions': deque([x,y],...), 'timestamps': deque(float,...)}}
        self.trajectories: dict = {}
        logger.info("TrajectoryPredictor initialised: window=%d, horizons=%s s",
                    window_size, list(PRED_HORIZONS))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_trajectory(self, global_id: int, position: list,
                          timestamp: float = None):
        """
        Record a new position observation.

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
        return self._time_based_prediction(
            list(traj['positions']),
            list(traj['timestamps']),
        )

    def predict_list(self, global_id: int) -> list | None:
        """
        Predicted positions as a plain list of [x, y] points.
        Preserves the interface expected by AlertSystem.check_alerts().
        """
        pred = self.predict(global_id)
        return list(pred.values()) if pred else None

    def get_velocity(self, global_id: int):
        """Return current velocity [vx, vy] in px/s, or None."""
        traj = self.trajectories.get(global_id)
        if not traj or len(traj['positions']) < 2:
            return None
        pos_arr = np.array(list(traj['positions']))
        t_arr   = np.array(list(traj['timestamps']))
        dt  = np.diff(t_arr)
        dxy = np.diff(pos_arr, axis=0)
        mask = dt > 1e-6
        if not mask.any():
            return None
        vels    = dxy[mask] / dt[mask, None]             # (M, 2) px/s
        weights = np.exp(np.linspace(-2, 0, len(vels)))  # recent points weighted higher
        return np.average(vels, axis=0, weights=weights)

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
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _time_based_prediction(self, positions: list, timestamps: list) -> dict | None:
        """
        Compute exponentially-weighted velocity from all stored observations,
        then project forward at each horizon in PRED_HORIZONS.
        """
        pos_arr = np.array(positions)   # (N, 2)
        t_arr   = np.array(timestamps)  # (N,)

        dt  = np.diff(t_arr)            # (N-1,)
        dxy = np.diff(pos_arr, axis=0)  # (N-1, 2)

        mask = dt > 1e-6                # guard against duplicate timestamps
        if not mask.any():
            return None

        vels     = dxy[mask] / dt[mask, None]          # px/s, shape (M, 2)
        weights  = np.exp(np.linspace(-2, 0, len(vels)))  # exp decay, newest = 1.0
        velocity = np.average(vels, axis=0, weights=weights)  # [vx, vy] px/s

        last_pos = pos_arr[-1]
        return {f't+{h}s': (last_pos + velocity * h).tolist() for h in PRED_HORIZONS}
