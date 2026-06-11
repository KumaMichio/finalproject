"""
Object Detection Module
Uses YOLOv8 for detecting vehicles and pedestrians in camera frames
"""

from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Object detector using YOLOv8/YOLO11 for real-time detection."""

    # COCO class indices of interest (used by stock 'yolov8*'/'yolo11*' checkpoints)
    CLASSES_COCO = {
        0: 'person',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }

    # Class indices for the VN-traffic fine-tuned checkpoint (5 merged classes,
    # see custom_tracking_system/data/visdrone_vn.yaml for the VisDrone -> these
    # class merge mapping used during fine-tuning)
    CLASSES_VN = {
        0: 'person',
        1: 'car',
        2: 'motorcycle',
        3: 'bus',
        4: 'truck',
    }

    CLASS_COLORS = {
        'person':     (255,   0,   0),
        'car':        (  0, 255,   0),
        'motorcycle': (  0, 255, 255),
        'bus':        (  0,   0, 255),
        'truck':      (255, 255,   0),
    }

    def __init__(self, model_type: str = 'yolov8s', conf_threshold: float = 0.35,
                 device: str | None = None, half: bool = False,
                 class_map: dict | None = None):
        """
        Args:
            model_type: YOLO variant/checkpoint. Either a stock pretrained alias
                ('yolov8n', 'yolov8s', 'yolov8m', 'yolo11n', 'yolo11s', 'yolo11m', ...)
                — resolved to '<model_type>.pt' and downloaded by ultralytics — or a
                path to a custom fine-tuned checkpoint ending in '.pt'
                (e.g. 'weights/yolo11m_vn.pt').
            conf_threshold: Minimum confidence to keep a detection
            device: 'cpu', 'cuda', '0', etc. None = auto
            half: FP16 inference — saves ~800 MB VRAM, use when running alongside CARLA
            class_map: optional override of {class_id: class_name}. If None,
                auto-selected: CLASSES_VN for a custom '.pt' checkpoint path,
                CLASSES_COCO for a stock pretrained alias.
        """
        import torch
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device or ('0' if torch.cuda.is_available() else 'cpu')
        self.half = half
        self.model = None
        if class_map is not None:
            self.CLASSES = class_map
        elif model_type.endswith('.pt'):
            self.CLASSES = self.CLASSES_VN
        else:
            self.CLASSES = self.CLASSES_COCO
        self._load_model()
        logger.info("ObjectDetector initialised: %s on %s%s",
                    model_type, self.device, " (FP16)" if half else "")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        from ultralytics import YOLO
        weights = self.model_type if self.model_type.endswith('.pt') else f'{self.model_type}.pt'
        try:
            self.model = YOLO(weights)
            if self.half:
                self.model.model.half()
            logger.info("Model loaded: %s", weights)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    def _parse_results(self, results, frame_shape=None) -> list[dict]:
        """Convert a single YOLOv8 Results object to the canonical detection list."""
        detections = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            if cls_id not in self.CLASSES:
                continue
            detections.append({
                'box':        [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class':      self.CLASSES[cls_id],
                'class_id':   cls_id,
            })
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect objects in a single frame.

        Args:
            frame: H×W×3 numpy array (BGR or RGB — YOLOv8 handles both)

        Returns:
            list of {'box': [x1,y1,x2,y2], 'confidence': float,
                     'class': str, 'class_id': int}
        """
        if self.model is None:
            return []
        try:
            results = self.model(
                frame,
                imgsz=640,
                conf=self.conf_threshold,
                classes=list(self.CLASSES.keys()),
                device=self.device,
                verbose=False,
            )
            return self._parse_results(results[0])
        except Exception as exc:
            logger.error("Detection error: %s", exc)
            return []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """
        Detect objects in a batch of frames (one forward pass for all cameras).

        Args:
            frames: list of H×W×3 arrays, one per camera

        Returns:
            list[list[dict]] — detections[i] corresponds to frames[i]
        """
        if self.model is None or not frames:
            return [[] for _ in frames]
        try:
            results = self.model(
                frames,
                imgsz=640,
                conf=self.conf_threshold,
                classes=list(self.CLASSES.keys()),
                device=self.device,
                verbose=False,
            )
            return [self._parse_results(r) for r in results]
        except Exception as exc:
            logger.error("Batch detection error: %s", exc)
            return [[] for _ in frames]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, frame: np.ndarray, detections: list[dict],
                  show_labels: bool = True) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = self.CLASS_COLORS.get(det['class'], (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        return {
            'model_type':     self.model_type,
            'device':         self.device,
            'half':           self.half,
            'conf_threshold': self.conf_threshold,
            'classes':        self.CLASSES,
            'status':         'loaded' if self.model else 'not_loaded',
        }
