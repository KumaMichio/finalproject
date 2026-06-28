#!/usr/bin/env python3
"""
Merge two or more YOLO-format detection datasets that share the same 5 VN
classes (person, car, motorcycle, bus, truck — see prepare_visdrone_dataset.py
and collect_carla_detection_data.py) into a single dataset for fine-tuning
YOLO11m with train_yolo_detector.py.

Each input dataset must already follow the layout:
    <root>/images/{train,val}/*.jpg
    <root>/labels/{train,val}/*.txt

Files are copied (not moved) into the output dataset, prefixed with the
source name to avoid filename collisions.

Usage:
    python merge_detection_datasets.py \
        --datasets visdrone=data/visdrone_vn carla=data/carla_det \
        --out data/visdrone_carla_vn
"""

import argparse
import random
import shutil
from pathlib import Path

VN_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']

# Datasets without a pre-existing images/{train,val} split (flat images/ +
# labels/ instead) get this deterministic split applied on the fly, so they
# can still be merged without mutating the source directory.
FLAT_VAL_RATIO = 0.15
FLAT_SPLIT_SEED = 42


def _flat_split_stems(src_root: Path) -> dict[str, list[str]]:
    images_dir = src_root / 'images'
    labels_dir = src_root / 'labels'
    stems = [
        p.stem for p in sorted(images_dir.glob('*.jpg'))
        if (labels_dir / (p.stem + '.txt')).exists()
    ]
    rng = random.Random(FLAT_SPLIT_SEED)
    rng.shuffle(stems)
    n_val = int(len(stems) * FLAT_VAL_RATIO)
    val_stems = set(stems[:n_val])
    return {
        'val': sorted(val_stems),
        'train': sorted(s for s in stems if s not in val_stems),
    }


def merge_split(name: str, src_root: Path, out_root: Path, split: str) -> int:
    src_images = src_root / 'images' / split
    src_labels = src_root / 'labels' / split

    out_images = out_root / 'images' / split
    out_labels = out_root / 'labels' / split

    if src_images.exists():
        stems = [
            p.stem for p in sorted(src_images.glob('*.jpg'))
            if (src_labels / (p.stem + '.txt')).exists()
        ]
        flat_images, flat_labels = src_images, src_labels
    else:
        # No images/{train,val} subdirs — fall back to a flat images/ +
        # labels/ layout and carve out a deterministic split ourselves
        # (e.g. data/auto_label, which has no train/val split on disk).
        flat_images, flat_labels = src_root / 'images', src_root / 'labels'
        if not flat_images.exists():
            return 0
        stems = _flat_split_stems(src_root)[split]

    if not stems:
        return 0

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n = 0
    for stem in stems:
        dest_stem = f"{name}_{stem}"
        shutil.copy(flat_images / f"{stem}.jpg", out_images / f"{dest_stem}.jpg")
        shutil.copy(flat_labels / f"{stem}.txt", out_labels / f"{dest_stem}.txt")
        n += 1

    return n


def write_yaml(out_root: Path, yaml_path: Path, names: list[str]):
    lines = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        f"names: {names}",
        "",
    ]
    yaml_path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', required=True,
                         help="One or more name=path pairs, e.g. "
                              "visdrone=data/visdrone_vn carla=data/carla_det")
    parser.add_argument('--out', default='data/visdrone_carla_vn',
                         help='output dataset directory')
    parser.add_argument('--yaml', default=None,
                         help='path to write the ultralytics dataset config '
                              '(default: <out>.yaml)')
    parser.add_argument('--names', nargs='+', default=None,
                         help='class names, in id order (default: the 5 VN '
                              'traffic classes — override for datasets with a '
                              'different class set, e.g. --names 1 2 3 ... 0 '
                              'for plate_chars_* or --names person car '
                              'motorcycle bus truck license_plate for '
                              'plate_pseudolabel)')
    args = parser.parse_args()

    out_root = Path(args.out)
    yaml_path = Path(args.yaml) if args.yaml else Path(str(out_root) + '.yaml')
    names = args.names if args.names else VN_CLASS_NAMES

    for split in ('train', 'val'):
        total = 0
        for entry in args.datasets:
            if '=' not in entry:
                raise SystemExit(f"--datasets entries must be name=path, got: {entry}")
            name, src = entry.split('=', 1)
            n = merge_split(name, Path(src), out_root, split)
            print(f"  [{split}] {name}: {n} images")
            total += n
        print(f"[{split}] total: {total} images -> {out_root / 'images' / split}")

    write_yaml(out_root, yaml_path, names)
    print(f"\nDataset config written to {yaml_path}")
    print(f"You can now run: python train_yolo_detector.py --data {yaml_path}")


if __name__ == '__main__':
    main()
