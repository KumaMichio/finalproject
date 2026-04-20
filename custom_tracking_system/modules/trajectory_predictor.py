"""
Trajectory Prediction Module
Predicts future object positions based on historical movement
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class TrajectoryPredictor:
    """
    Predicts object trajectories using simple motion models
    """

    def __init__(self, window_size=10, pred_steps=5):
        """
        Initialize trajectory predictor

        Args:
            window_size: Number of past positions to use for prediction
            pred_steps: Number of future steps to predict
        """
        self.window_size = window_size
        self.pred_steps = pred_steps

        # Trajectories: {global_id: {'positions': deque, 'frames': deque}}
        self.trajectories = {}

        logger.info(f"TrajectoryPredictor initialized: window={window_size}, steps={pred_steps}")

    def update_trajectory(self, global_id, position, frame_idx):
        """
        Update trajectory for an object

        Args:
            global_id: Global object ID
            position: [x, y] - current position
            frame_idx: Current frame index
        """
        if global_id not in self.trajectories:
            self.trajectories[global_id] = {
                'positions': deque(maxlen=self.window_size),
                'frames': deque(maxlen=self.window_size)
            }

        self.trajectories[global_id]['positions'].append(position)
        self.trajectories[global_id]['frames'].append(frame_idx)

    def predict(self, global_id):
        """
        Predict future positions for an object

        Args:
            global_id: Global object ID

        Returns:
            list: predicted positions [[x, y], ...] or None
        """
        if global_id not in self.trajectories:
            return None

        traj_data = self.trajectories[global_id]
        positions = list(traj_data['positions'])
        frames = list(traj_data['frames'])

        if len(positions) < 2:
            return None

        # Use linear motion model
        predicted_positions = self._linear_prediction(positions, frames)

        return predicted_positions

    def _linear_prediction(self, positions, frames):
        """
        Linear motion prediction

        Args:
            positions: list of [x, y] positions
            frames: corresponding frame indices

        Returns:
            list: predicted positions
        """
        if len(positions) < 2:
            return None

        # Calculate velocity (difference between last two positions)
        pos_array = np.array(positions)
        last_pos = pos_array[-1]
        prev_pos = pos_array[-2]

        velocity = last_pos - prev_pos

        # Predict future positions
        predicted = []
        for step in range(1, self.pred_steps + 1):
            next_pos = last_pos + velocity * step
            predicted.append(next_pos.tolist())

        return predicted

    def predict_multiple_steps(self, global_id, steps=None):
        """
        Predict multiple future steps

        Args:
            global_id: Global object ID
            steps: Number of steps to predict (default: self.pred_steps)

        Returns:
            list: predicted trajectory
        """
        steps = steps or self.pred_steps
        original_steps = self.pred_steps

        # Temporarily change prediction steps
        self.pred_steps = steps
        prediction = self.predict(global_id)
        self.pred_steps = original_steps

        return prediction

    def get_trajectory_history(self, global_id):
        """
        Get historical trajectory

        Args:
            global_id: Global object ID

        Returns:
            dict: trajectory data
        """
        return self.trajectories.get(global_id, {'positions': [], 'frames': []})

    def get_velocity(self, global_id):
        """
        Calculate current velocity

        Args:
            global_id: Global object ID

        Returns:
            numpy array: [vx, vy] or None
        """
        traj_data = self.trajectories.get(global_id)
        if not traj_data or len(traj_data['positions']) < 2:
            return None

        positions = list(traj_data['positions'])
        pos_array = np.array(positions)

        # Calculate velocity from recent positions
        if len(pos_array) >= 3:
            # Use average of last few velocities for stability
            velocities = []
            for i in range(len(pos_array) - 1):
                vel = pos_array[i + 1] - pos_array[i]
                velocities.append(vel)

            return np.mean(velocities, axis=0)
        else:
            # Simple velocity
            return pos_array[-1] - pos_array[-2]

    def get_speed(self, global_id):
        """
        Calculate current speed

        Args:
            global_id: Global object ID

        Returns:
            float: speed (pixels per frame) or None
        """
        velocity = self.get_velocity(global_id)
        if velocity is None:
            return None

        return np.linalg.norm(velocity)

    def clear_trajectory(self, global_id):
        """
        Clear trajectory for an object

        Args:
            global_id: Global object ID
        """
        if global_id in self.trajectories:
            del self.trajectories[global_id]

    def get_all_trajectories(self):
        """
        Get all stored trajectories

        Returns:
            dict: all trajectories
        """
        return self.trajectories.copy()

    def get_statistics(self):
        """
        Get trajectory statistics

        Returns:
            dict: statistics
        """
        total_objects = len(self.trajectories)
        avg_trajectory_length = 0

        if total_objects > 0:
            lengths = [len(traj['positions']) for traj in self.trajectories.values()]
            avg_trajectory_length = np.mean(lengths)

        return {
            'total_objects': total_objects,
            'avg_trajectory_length': avg_trajectory_length,
            'window_size': self.window_size,
            'pred_steps': self.pred_steps
        }