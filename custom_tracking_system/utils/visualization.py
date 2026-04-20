"""
Visualization Utilities
Handles display and visualization of tracking results
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualizes tracking results on camera frames
    """

    def __init__(self):
        # Colors for different classes
        self.class_colors = {
            'person': (255, 0, 0),    # Blue
            'car': (0, 255, 0),       # Green
            'bus': (0, 0, 255),       # Red
            'truck': (255, 255, 0),   # Cyan
            'unknown': (128, 128, 128) # Gray
        }

        # Generate colors for global IDs
        self.id_colors = {}
        self._next_color_idx = 0

        logger.info("Visualizer initialized")

    def draw_tracks(self, frame, global_tracks, show_trajectories=False):
        """
        Draw tracked objects with global IDs

        Args:
            frame: numpy array (H, W, 3)
            global_tracks: list of track dicts
            show_trajectories: whether to draw trajectory lines

        Returns:
            numpy array: frame with visualizations
        """
        frame_copy = frame.copy()

        for track in global_tracks:
            global_id = track['global_id']
            box = track['box']
            class_name = track.get('class', 'unknown')

            # Get color for this ID
            color = self._get_id_color(global_id)

            x1, y1, x2, y2 = box

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{global_id} {class_name}"
            cv2.putText(frame_copy, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_copy

    def draw_trajectories(self, frame, trajectories, camera_id=None):
        """
        Draw historical trajectories

        Args:
            frame: numpy array (H, W, 3)
            trajectories: dict of {global_id: trajectory_data}
            camera_id: specific camera to show (None for all)

        Returns:
            numpy array: frame with trajectories
        """
        frame_copy = frame.copy()

        for global_id, traj_data in trajectories.items():
            positions = traj_data.get('positions', [])

            if len(positions) > 1:
                color = self._get_id_color(global_id)

                # Draw trajectory line
                for i in range(len(positions) - 1):
                    pt1 = tuple(map(int, positions[i]))
                    pt2 = tuple(map(int, positions[i + 1]))
                    cv2.line(frame_copy, pt1, pt2, color, 1)

                # Draw current position marker
                if positions:
                    current_pos = tuple(map(int, positions[-1]))
                    cv2.circle(frame_copy, current_pos, 3, color, -1)

        return frame_copy

    def draw_predictions(self, frame, predictions):
        """
        Draw predicted future positions

        Args:
            frame: numpy array (H, W, 3)
            predictions: dict of {global_id: predicted_positions}

        Returns:
            numpy array: frame with predictions
        """
        frame_copy = frame.copy()

        for global_id, pred_positions in predictions.items():
            if not pred_positions:
                continue

            color = self._get_id_color(global_id)

            # Draw predicted path
            for i in range(len(pred_positions) - 1):
                pt1 = tuple(map(int, pred_positions[i]))
                pt2 = tuple(map(int, pred_positions[i + 1]))
                cv2.line(frame_copy, pt1, pt2, color, 1)

            # Draw prediction markers
            for pos in pred_positions:
                pt = tuple(map(int, pos))
                cv2.circle(frame_copy, pt, 2, color, -1)

        return frame_copy

    def draw_rois(self, frame, rois):
        """
        Draw regions of interest

        Args:
            frame: numpy array (H, W, 3)
            rois: list of {'name': str, 'polygon': [[x,y], ...]}

        Returns:
            numpy array: frame with ROIs
        """
        frame_copy = frame.copy()

        for roi in rois:
            name = roi['name']
            polygon = roi['polygon']

            if len(polygon) < 3:
                continue

            # Convert to numpy array
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw polygon
            cv2.polylines(frame_copy, [pts], True, (0, 255, 255), 2)

            # Draw label
            if polygon:
                text_pos = tuple(map(int, polygon[0]))
                cv2.putText(frame_copy, name, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame_copy

    def draw_alerts(self, frame, alerts):
        """
        Draw active alerts

        Args:
            frame: numpy array (H, W, 3)
            alerts: list of alert dicts

        Returns:
            numpy array: frame with alerts
        """
        frame_copy = frame.copy()

        for alert in alerts:
            global_id = alert['global_id']
            roi_name = alert['roi_name']
            eta = alert['eta_frames']

            color = (0, 0, 255)  # Red for alerts

            # Draw alert text
            alert_text = f"ALERT: ID{global_id} -> {roi_name} ({eta}f)"
            cv2.putText(frame_copy, alert_text, (10, 30 + global_id * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame_copy

    def create_multi_camera_display(self, frames_dict, cols=2):
        """
        Create a multi-camera display grid

        Args:
            frames_dict: dict of {camera_id: frame}
            cols: number of columns in grid

        Returns:
            numpy array: combined display frame
        """
        if not frames_dict:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frames = list(frames_dict.values())
        camera_ids = list(frames_dict.keys())

        # Calculate grid dimensions
        n_frames = len(frames)
        rows = (n_frames + cols - 1) // cols

        # Get frame dimensions (assume all same size)
        h, w = frames[0].shape[:2]

        # Create grid
        grid_h = rows * h
        grid_w = cols * w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols

            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w

            grid[y1:y2, x1:x2] = frame

            # Add camera label
            cv2.putText(grid, camera_ids[i], (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return grid

    def _get_id_color(self, global_id):
        """
        Get consistent color for global ID

        Args:
            global_id: Global object ID

        Returns:
            tuple: (B, G, R) color
        """
        if global_id not in self.id_colors:
            # Generate new color
            np.random.seed(global_id)
            color = tuple(map(int, np.random.rand(3) * 256))
            self.id_colors[global_id] = color

        return self.id_colors[global_id]

    def save_frame(self, frame, filename):
        """
        Save frame to file

        Args:
            frame: numpy array
            filename: output filename
        """
        try:
            cv2.imwrite(filename, frame)
            logger.debug(f"Frame saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")

    def create_video_writer(self, filename, fps=10, frame_size=(1920, 1080)):
        """
        Create video writer

        Args:
            filename: output video file
            fps: frames per second
            frame_size: (width, height)

        Returns:
            cv2.VideoWriter: video writer object
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        return writer