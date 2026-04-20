"""
Object Detection Module
Uses YOLOv5 for detecting vehicles and pedestrians in camera frames
"""

import torch
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Object detector using YOLOv5 for real-time detection
    """

    def __init__(self, model_type='yolov5s', conf_threshold=0.4, device=None):
        """
        Initialize object detector

        Args:
            model_type: YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            conf_threshold: Confidence threshold for detections
            device: Device to run model on ('cpu', 'cuda', None for auto)
        """
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Classes of interest (COCO dataset indices)
        self.classes_of_interest = {
            0: 'person',      # person
            2: 'car',         # car
            5: 'bus',         # bus
            7: 'truck'        # truck
        }

        self.model = None
        self._load_model()

        logger.info(f"ObjectDetector initialized with {model_type} on {self.device}")

    def _load_model(self):
        """Load YOLOv5 model"""
        import os, sys
        try:
            # Set hub cache on same drive as project (avoids Windows cross-drive path error)
            hub_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'models', 'hub')
            os.makedirs(hub_dir, exist_ok=True)
            torch.hub.set_dir(hub_dir)

            # YOLOv5 uses bare `utils` imports that conflict with our project's utils/ package.
            # Temporarily remove our utils from sys.modules so YOLOv5 finds its own.
            _saved = {k: sys.modules.pop(k)
                      for k in list(sys.modules)
                      if k == 'utils' or k.startswith('utils.')}
            try:
                self.model = torch.hub.load('ultralytics/yolov5:v6.1', self.model_type,
                                            pretrained=True, force_reload=False)
            finally:
                # Remove YOLOv5's utils from cache, restore our utils
                for k in [k for k in list(sys.modules)
                          if k == 'utils' or k.startswith('utils.')]:
                    del sys.modules[k]
                sys.modules.update(_saved)

            self.model.to(self.device)
            self.model.conf = self.conf_threshold
            self.model.classes = list(self.classes_of_interest.keys())

            logger.info(f"Model loaded successfully: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame):
        """
        Detect objects in frame

        Args:
            frame: numpy array (H, W, 3) - RGB image

        Returns:
            list: [{'box': [x1,y1,x2,y2], 'confidence': float, 'class': str, 'class_id': int}]
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            # Run inference
            results = self.model(frame)

            # Parse results
            detections = []
            for det in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2, conf, cls = det

                class_id = int(cls)
                if class_id in self.classes_of_interest:
                    detection = {
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': self.classes_of_interest[class_id],
                        'class_id': class_id
                    }
                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def visualize(self, frame, detections, show_labels=True):
        """
        Draw bounding boxes on frame

        Args:
            frame: numpy array (H, W, 3)
            detections: list of detection dicts
            show_labels: whether to show class labels

        Returns:
            numpy array: frame with bounding boxes
        """
        frame_copy = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            class_name = det['class']

            # Choose color based on class
            color = self._get_class_color(class_name)

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                # Draw label
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame_copy, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame_copy

    def _get_class_color(self, class_name):
        """Get color for class visualization"""
        colors = {
            'person': (255, 0, 0),    # Blue
            'car': (0, 255, 0),       # Green
            'bus': (0, 0, 255),       # Red
            'truck': (255, 255, 0)    # Cyan
        }
        return colors.get(class_name, (255, 255, 255))

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'status': 'not_loaded'}

        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'conf_threshold': self.conf_threshold,
            'classes': self.classes_of_interest,
            'status': 'loaded'
        }