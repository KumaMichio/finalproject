"""
Fine-tune YOLO11m on a VN-traffic detection dataset (5 classes:
person, car, motorcycle, bus, truck).

Two data sources, both producing the same images/{train,val} +
labels/{train,val} YOLO layout with these 5 classes:

    1. prepare_visdrone_dataset.py  -> data/visdrone_vn/   (real-world, VisDrone)
    2. collect_carla_detection_data.py -> data/carla_det/  (synthetic, CARLA CCTV)

Combine them with merge_detection_datasets.py before training:

    python prepare_visdrone_dataset.py --out data/visdrone_vn
    python collect_carla_detection_data.py --out data/carla_det   # run inside CARLA
    python merge_detection_datasets.py \
        --datasets visdrone=data/visdrone_vn carla=data/carla_det \
        --out data/visdrone_carla_vn

Usage:
    python train_yolo_detector.py --data data/visdrone_carla_vn.yaml --epochs 50 \
        --batch 16 --out weights/yolo11m_vn.pt

(VisDrone-only is also fine: --data data/visdrone_vn.yaml)

Hardware notes:
    - RTX 4060 8GB:  --batch 16-24 --device 0
    - GTX 1050Ti 4GB: --batch 4-8 --device 0 --half
    - CPU only:      --device cpu --batch 4 --epochs 2  (smoke test only)

The output checkpoint (weights/yolo11m_vn.pt by default) is picked up
automatically by main.py / ObjectDetector if present.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/visdrone_carla_vn.yaml',
                         help='ultralytics dataset yaml — see merge_detection_datasets.py '
                              '(use data/visdrone_vn.yaml for VisDrone-only)')
    parser.add_argument('--model', default='yolo11m.pt',
                         help='base checkpoint to fine-tune from')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default=None,
                         help="'0' for first GPU, 'cpu', etc. None = auto")
    parser.add_argument('--half', action='store_true',
                         help='enable AMP mixed-precision training (saves VRAM)')
    parser.add_argument('--workers', type=int, default=8,
                         help='dataloader worker processes (lower if CPU RAM-limited)')
    parser.add_argument('--out', default='weights/yolo11m_vn.pt')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='yolo11m_vn')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=args.half,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )

    best = Path(args.project) / args.name / 'weights' / 'best.pt'
    out = Path(args.out)
    if best.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out)
        print(f"\nSaved fine-tuned checkpoint to {out}")
    else:
        print(f"\nWARNING: expected {best} not found; check the runs/ directory")


if __name__ == '__main__':
    main()
