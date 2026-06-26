"""
Incident Snapshot Logger — opt-in, NOT used in production.

Saves one annotated JPEG + one JSON (incident type/severity/details, label
left blank for hand-labeling) per incident, for any severity — not just
CRITICAL like EvidencePackage's 30s video clips. Used to build a true/false
positive labeled set for tuning IncidentDetector thresholds; see
docs/plan_tinh_chinh_nguong_incident.md.
"""

import json
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class IncidentSnapshotLogger:
    def __init__(self, output_dir: str, visualizer):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = visualizer
        logger.info("IncidentSnapshotLogger: saving snapshots to %s", self.output_dir)

    def log(self, incident: dict, frame, global_tracks: list):
        """
        Args:
            incident: one incident dict as returned by IncidentDetector.update()
            frame: current camera frame (RGB, H x W x 3) for incident['camera_id']
            global_tracks: full global_tracks list for that frame/camera, used
                to draw all boxes+IDs so the reviewer can find the relevant
                object(s) by the ID printed in incident['message']
        """
        ts = incident['timestamp'].strftime('%Y%m%d_%H%M%S_%f')
        incident_id = f"{incident['type']}_{incident['global_id']}_{ts}"

        annotated_rgb = self.visualizer.draw_tracks(frame, global_tracks)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        img_path = self.output_dir / f"{incident_id}.jpg"
        cv2.imwrite(str(img_path), annotated_bgr)

        meta = {
            'incident_id': incident_id,
            'type': incident['type'],
            'severity': incident['severity'],
            'global_id': incident['global_id'],
            'camera_id': incident['camera_id'],
            'message': incident['message'],
            'details': incident['details'],
            'timestamp': incident['timestamp'].isoformat(),
            'label': None,  # fill by hand: "tp" or "fp" (xem Bước 2 trong plan)
        }
        meta_path = self.output_dir / f"{incident_id}.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding='utf-8')
