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

    def draw_collision_banner(self, frame):
        """Vẽ banner đỏ "VA CHẠM!" + viền đỏ quanh frame — hiển thị vài giây
        sau khi ScenarioController ghi nhận 1 va chạm trên camera này, để
        người xem dễ nhận ra sự kiện đang xảy ra trên dashboard."""
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        cv2.rectangle(frame_copy, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        text = "VA CHAM!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.2, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        cv2.rectangle(frame_copy, (x - 10, 10), (x + tw + 10, 10 + th + 20),
                       (0, 0, 255), -1)
        cv2.putText(frame_copy, text, (x, 10 + th + 5), font, scale,
                     (255, 255, 255), thickness)
        return frame_copy

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

    # Màu cố định (vàng, BGR) cho mọi đường dự đoán — phân biệt rõ với màu
    # bounding box (theo từng global_id) để dễ nhận ra khi demo.
    PREDICTION_COLOR = (0, 255, 255)

    def draw_predictions(self, frame, predictions, global_tracks=None):
        """
        Draw predicted future trajectory (dashed line + markers) starting
        from the object's current position.

        Args:
            frame: numpy array (H, W, 3)
            predictions: dict {global_id: [[x,y], ...]} — vị trí dự đoán tại
                các horizon (t+0.5s, t+1s, ...), thứ tự tăng dần theo thời gian.
            global_tracks: optional list track dicts (để vẽ đoạn nối từ vị trí
                HIỆN TẠI của object tới điểm dự đoán đầu tiên).

        Returns:
            numpy array: frame with predicted trajectories overlaid
        """
        frame_copy = frame.copy()
        color = self.PREDICTION_COLOR

        centers = {}
        if global_tracks:
            for t in global_tracks:
                box = t['box']
                centers[t['global_id']] = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

        for global_id, pred_positions in predictions.items():
            if not pred_positions:
                continue

            points = []
            if global_id in centers:
                points.append(centers[global_id])
            points += [tuple(map(int, p)) for p in pred_positions]
            if len(points) < 2:
                continue

            # Đường nét đứt nối vị trí hiện tại -> các điểm dự đoán tương lai
            for i in range(len(points) - 1):
                self._draw_dashed_line(frame_copy, points[i], points[i + 1], color)

            # Marker tại mỗi điểm dự đoán (không tính điểm hiện tại)
            for pt in points[1:]:
                cv2.circle(frame_copy, pt, 3, color, -1)

            # Nhãn "predicted" tại điểm dự đoán xa nhất
            last = points[-1]
            cv2.putText(frame_copy, "predicted", (last[0] + 5, last[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame_copy

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness=2, dash_len=8):
        """Vẽ đoạn thẳng nét đứt từ pt1 đến pt2."""
        dist = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        if dist < 1:
            return
        n_dashes = max(1, int(dist / dash_len))
        for i in range(n_dashes):
            start = i / n_dashes
            end = min((i + 0.5) / n_dashes, 1.0)
            x1 = int(pt1[0] + (pt2[0] - pt1[0]) * start)
            y1 = int(pt1[1] + (pt2[1] - pt1[1]) * start)
            x2 = int(pt1[0] + (pt2[0] - pt1[0]) * end)
            y2 = int(pt1[1] + (pt2[1] - pt1[1]) * end)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

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