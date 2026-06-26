#!/usr/bin/env python3
"""
Smoke test for DualReIDExtractor (modules/reid.py).

Confirms the already-wired DualReIDExtractor in main.py actually works at
runtime: vehicle weights load, person crops route through the Market-1501
model, vehicle crops route through the VeRi-776 fine-tuned model, and the two
embedding spaces are kept separate (different calibration group).

Usage:
    python scripts/eval/smoke_test_dual_reid.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.reid import DualReIDExtractor  # noqa: E402

CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']
AUTO_LABEL_DIR = project_root / 'data' / 'auto_label'


def find_crop(target_class_id: int):
    """Scan label files for the first box of `target_class_id`, return (image_path, xyxy_box)."""
    labels_dir = AUTO_LABEL_DIR / 'labels'
    images_dir = AUTO_LABEL_DIR / 'images'

    for label_path in sorted(labels_dir.glob('*.txt'))[:200]:
        lines = label_path.read_text(encoding='utf-8').strip().splitlines()
        for line in lines:
            parts = line.split()
            if not parts or int(parts[0]) != target_class_id:
                continue
            cls, cx, cy, w, h = (float(p) for p in parts[:5])
            img_path = images_dir / (label_path.stem + '.jpg')
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                iw, ih = im.size
            x1 = (cx - w / 2) * iw
            y1 = (cy - h / 2) * ih
            x2 = (cx + w / 2) * iw
            y2 = (cy + h / 2) * ih
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue  # too small to crop reliably
            return img_path, [x1, y1, x2, y2]
    return None, None


def load_frame_rgb(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        return np.array(im.convert('RGB'))


def main():
    vehicle_weights = project_root / 'weights' / 'osnet_veri776_v2.pth'
    print(f"[1] Checking vehicle weights file: {vehicle_weights}")
    assert vehicle_weights.exists(), f"Missing weights file: {vehicle_weights}"
    print("    OK — file exists "
          f"({vehicle_weights.stat().st_size / 1e6:.1f} MB)")

    print("[2] Instantiating DualReIDExtractor ...")
    reid = DualReIDExtractor(vehicle_weights=str(vehicle_weights))
    info = reid.get_model_info()
    print(f"    model_info = {info}")
    assert info['vehicle_model'] != 'NOT LOADED (fallback to person model)', \
        "Vehicle model did not load — check logs above"
    assert reid._vehicle_model is not None, "self._vehicle_model is still None"
    print("    OK — vehicle model loaded (not falling back to person model)")

    print("[3] Finding a real person crop and a real car crop from data/auto_label ...")
    person_img, person_box = find_crop(CLASS_NAMES.index('person'))
    car_img, car_box = find_crop(CLASS_NAMES.index('car'))
    assert person_img is not None, "Could not find a person box in the first 200 label files"
    assert car_img is not None, "Could not find a car box in the first 200 label files"
    print(f"    person crop: {person_img.name} box={[round(v, 1) for v in person_box]}")
    print(f"    car crop:    {car_img.name} box={[round(v, 1) for v in car_box]}")

    print("[4] Extracting features ...")
    person_frame = load_frame_rgb(person_img)
    car_frame = load_frame_rgb(car_img)

    person_feat = reid.extract_feature(person_frame, person_box, object_class='person')
    car_feat = reid.extract_feature(car_frame, car_box, object_class='car')

    assert person_feat is not None, "extract_feature returned None for person crop"
    assert car_feat is not None, "extract_feature returned None for car crop"
    print(f"    person feature shape={person_feat.shape}, "
          f"norm={np.linalg.norm(person_feat):.4f}")
    print(f"    car feature shape={car_feat.shape}, "
          f"norm={np.linalg.norm(car_feat):.4f}")
    assert person_feat.shape == (1, reid.feature_dim)
    assert car_feat.shape == (1, reid.feature_dim)

    print("[5] Verifying person/vehicle route to different models + groups ...")
    assert DualReIDExtractor.group_for_class('person') == 'person'
    assert DualReIDExtractor.group_for_class('car') == 'vehicle'
    # The class-conditional branch in DualReIDExtractor.extract_feature swaps
    # self.model between self._person_model and self._vehicle_model — confirm
    # those are in fact two distinct model objects (not silently the same one).
    assert reid._person_model is not reid._vehicle_model, \
        "Person and vehicle models are the same object — dual-model routing broken"
    print("    OK — group_for_class routes correctly, person/vehicle models are distinct objects")

    print("\nAll checks passed. DualReIDExtractor is wired and functional.")


if __name__ == '__main__':
    main()
