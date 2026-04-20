"""
Alert System Module
Generates alerts when objects enter regions of interest
"""

from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Monitors object trajectories and generates alerts for ROIs
    """

    def __init__(self, trajectory_predictor, alert_threshold=5):
        """
        Initialize alert system

        Args:
            trajectory_predictor: TrajectoryPredictor instance
            alert_threshold: Minimum frames before alert
        """
        self.trajectory_predictor = trajectory_predictor
        self.alert_threshold = alert_threshold

        # ROIs: {camera_id: [{'name': str, 'polygon': [[x,y], ...]}]}
        self.rois = {}

        # Alert history: {global_id: [alert_dicts]}
        self.alert_history = defaultdict(list)

        logger.info("AlertSystem initialized")

    def set_rois(self, rois):
        """
        Set regions of interest

        Args:
            rois: dict of ROIs per camera — supports both list format and {'zones': [...]} format
        """
        self.rois = {}
        for cam_id, roi_data in rois.items():
            if isinstance(roi_data, dict) and 'zones' in roi_data:
                self.rois[cam_id] = roi_data['zones']
            elif isinstance(roi_data, list):
                self.rois[cam_id] = roi_data
            else:
                self.rois[cam_id] = []
        total_zones = sum(len(zones) for zones in self.rois.values())
        logger.info(f"Set {total_zones} ROIs for {len(self.rois)} cameras")

    def check_alerts(self, global_id, camera_id, predicted_positions, current_box):
        """
        Check if object trajectory triggers alerts

        Args:
            global_id: Global object ID
            camera_id: Current camera ID
            predicted_positions: List of predicted [x, y] positions
            current_box: Current bounding box [x1,y1,x2,y2]

        Returns:
            list: alert dictionaries
        """
        alerts = []

        if predicted_positions is None:
            return alerts

        if camera_id not in self.rois:
            return alerts

        # Check each ROI for this camera
        for roi in self.rois[camera_id]:
            roi_name = roi['name']
            polygon = roi['polygon']

            # Estimate arrival time (None if trajectory never enters ROI)
            eta = self._estimate_arrival_time(predicted_positions, polygon)

            if eta is not None and eta <= self.alert_threshold:
                alert = {
                    'type': 'ROI_WARNING',
                    'global_id': global_id,
                    'camera_id': camera_id,
                    'roi_name': roi_name,
                    'timestamp': datetime.now(),
                    'eta_frames': eta,
                    'current_position': self._box_center(current_box),
                    'predicted_positions': predicted_positions
                }
                alerts.append(alert)

        return alerts

    def _will_enter_roi(self, predicted_positions, polygon):
        """
        Check if trajectory will enter ROI

        Args:
            predicted_positions: list of [x, y]
            polygon: list of [x, y] vertices

        Returns:
            bool: True if trajectory enters ROI
        """
        return self._estimate_arrival_time(predicted_positions, polygon) is not None

    def _point_in_polygon(self, point, polygon):
        """
        Ray casting algorithm to check if point is inside polygon

        Args:
            point: [x, y]
            polygon: list of [x, y] vertices

        Returns:
            bool: True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _estimate_arrival_time(self, predicted_positions, polygon):
        """
        Estimate frames until object enters ROI.

        Args:
            predicted_positions: list of [x, y]
            polygon: list of [x, y] vertices

        Returns:
            int: step index (1-based) of first entry, or None if never enters
        """
        for step, pos in enumerate(predicted_positions, 1):
            if self._point_in_polygon(pos, polygon):
                return step
        return None

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

    def log_alert(self, alert):
        """
        Log an alert

        Args:
            alert: alert dictionary
        """
        self.alert_history[alert['global_id']].append(alert)

        logger.warning(f"ALERT: Object {alert['global_id']} entering {alert['roi_name']} "
                      f"in camera {alert['camera_id']} (ETA: {alert['eta_frames']} frames)")

        # Print to console for immediate visibility
        print(f"🚨 ALERT: Object {alert['global_id']} will enter {alert['roi_name']} "
              f"in {alert['eta_frames']} frames!")

    def get_alert_history(self, global_id=None):
        """
        Get alert history

        Args:
            global_id: Specific object ID (None for all)

        Returns:
            dict or list: alert history
        """
        if global_id is not None:
            return self.alert_history.get(global_id, [])
        else:
            return dict(self.alert_history)

    def get_recent_alerts(self, minutes=5):
        """
        Get alerts from the last N minutes

        Args:
            minutes: Time window in minutes

        Returns:
            list: recent alerts
        """
        now = datetime.now()
        recent_alerts = []

        for alerts in self.alert_history.values():
            for alert in alerts:
                elapsed = (now - alert['timestamp']).total_seconds() / 60
                if elapsed <= minutes:
                    recent_alerts.append(alert)

        return recent_alerts

    def clear_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        logger.info("Alert history cleared")

    def get_statistics(self):
        """
        Get alert statistics

        Returns:
            dict: statistics
        """
        total_alerts = sum(len(alerts) for alerts in self.alert_history.values())
        unique_objects = len(self.alert_history)

        recent_alerts = self.get_recent_alerts(5)
        recent_count = len(recent_alerts)

        return {
            'total_alerts': total_alerts,
            'unique_objects': unique_objects,
            'recent_alerts_5min': recent_count,
            'rois_configured': sum(len(r) for r in self.rois.values())
        }