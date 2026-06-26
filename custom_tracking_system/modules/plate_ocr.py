"""
License Plate OCR Module — supplementary signal for cross-camera Re-ID.

This module does NOT replace ReID. It only reads a license-plate string from
a vehicle crop (when legible) so that `GlobalTracker` can VETO a ReID match
when two confidently-read plates clearly disagree. It never creates a match
on its own — see modules/global_tracking.py for how the veto is applied.

No model training is involved: this wraps a pretrained, general-purpose OCR
engine (EasyOCR). Vietnamese plates are digits + unaccented Latin letters
(not Vietnamese-language text), so a generic OCR model is expected to read
the characters when the plate region is legible; the real bottleneck is
image resolution/blur at typical traffic-camera distance, not language.

Usage:
    reader = PlateOCRReader()
    result = reader.read_plate(frame, box, object_class='car')
    # result: {'text': '29A12345', 'confidence': 0.83} or None
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Pure string helpers — no model/GPU required, safe to unit-test directly.
# ---------------------------------------------------------------------------

_NON_ALNUM_RE = re.compile(r'[^A-Z0-9]')


def normalize_plate_text(text: str) -> str:
    """Uppercase and strip everything but letters/digits (drop -, ., spaces)."""
    return _NON_ALNUM_RE.sub('', text.upper())


def is_plausible_vn_plate(text: str) -> bool:
    """Coarse shape check for a Vietnamese plate, on a normalized string.

    Vietnamese plates: 2-digit province code + 1-2 letters (series) +
    3-6 digits (registration number, often written as NNN.NN or NNNNN).
    This is deliberately permissive — it only filters out OCR garbage that
    clearly isn't plate-shaped, not a legal-format validator.
    """
    s = normalize_plate_text(text)
    if not (6 <= len(s) <= 9):
        return False
    if not s[:2].isdigit():
        return False

    i = 2
    n_letters = 0
    while i < len(s) and s[i].isalpha() and n_letters < 2:
        i += 1
        n_letters += 1
    if n_letters == 0:
        return False

    rest = s[i:]
    return rest.isdigit() and 3 <= len(rest) <= 6


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance, stdlib-only (no extra dependency)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def plate_similarity(a: str, b: str) -> int:
    """Edit distance between two plate strings after normalization.

    Lower is more similar (0 = identical after normalization). Tolerant of
    common single-character OCR confusions (0/O, 1/I, 8/B, ...) when the
    caller allows a small max edit distance (e.g. <=1).
    """
    return _edit_distance(normalize_plate_text(a), normalize_plate_text(b))


# ---------------------------------------------------------------------------
# OCR engine wrapper
# ---------------------------------------------------------------------------

class PlateOCRReader:
    """Reads a license-plate string from a vehicle bounding-box crop.

    Only EasyOCR is supported for now (see Chapter 3 of the report for the
    rationale: it reuses the project's existing torch/torchvision stack
    instead of pulling in a second deep-learning framework like PaddlePaddle).
    """

    def __init__(self, engine: str = 'easyocr', device: str = None,
                 min_confidence: float = 0.5, crop_region: str = 'bottom_half',
                 upscale: float = 2.5):
        """
        Args:
            engine:         OCR backend name. Only 'easyocr' implemented.
            device:         'cuda' | 'cpu' | None (auto-detect, same
                             convention as detector.py / reid.py).
            min_confidence: minimum OCR confidence to accept a raw text read.
            crop_region:    'full' | 'bottom_half' | 'bottom_third' — which
                             part of the vehicle bbox to feed to OCR (plates
                             sit low on the vehicle silhouette in elevated
                             CCTV views).
            upscale:        linear upscale factor applied to the crop before
                             OCR (plates are tiny at typical camera distance).
        """
        if engine != 'easyocr':
            raise ValueError(f"Unsupported OCR engine: {engine!r} (only 'easyocr')")

        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        self.crop_region = crop_region
        self.upscale = upscale

        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        logger.info(
            "PlateOCRReader ready: engine=easyocr, device=%s, crop_region=%s",
            self.device, self.crop_region,
        )

    def _crop_region(self, frame: np.ndarray, box) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            return None

        if self.crop_region == 'bottom_half':
            y1 = y1 + (y2 - y1) // 2
        elif self.crop_region == 'bottom_third':
            y1 = y1 + 2 * (y2 - y1) // 3
        # 'full' uses the box unchanged

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        return crop

    def read_plate(self, frame: np.ndarray, box, object_class: str) -> dict | None:
        """Try to read a plate from one detection.

        Returns {'text': str, 'confidence': float} (text already passed
        through `is_plausible_vn_plate`) or None — either because the class
        isn't a vehicle, the crop is degenerate, or no plausible plate text
        was found above `min_confidence`.
        """
        if object_class not in _VEHICLE_CLASSES:
            return None

        crop = self._crop_region(frame, box)
        if crop is None:
            return None

        if self.upscale and self.upscale != 1.0:
            import cv2
            crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale,
                               interpolation=cv2.INTER_CUBIC)

        try:
            results = self._reader.readtext(crop)
        except Exception:
            logger.exception("EasyOCR inference failed on a plate crop")
            return None

        best = None
        for _bbox, text, conf in results:
            if conf < self.min_confidence:
                continue
            if not is_plausible_vn_plate(text):
                continue
            if best is None or conf > best[1]:
                best = (normalize_plate_text(text), conf)

        if best is None:
            return None
        return {'text': best[0], 'confidence': float(best[1])}
