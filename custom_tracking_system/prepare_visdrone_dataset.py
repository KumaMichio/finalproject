"""
Download VisDrone2019-DET (train + val) and convert it into a YOLO-format
detection dataset for fine-tuning YOLO11m on VN-traffic-like scenes
(elevated/CCTV-angle, dense motorbikes/cars/pedestrians).

VisDrone's 10 categories are merged down to the 5 classes used by
ObjectDetector.CLASSES_VN (person, car, motorcycle, bus, truck):

    pedestrian, people  -> person
    car, van            -> car
    bicycle, motor      -> motorcycle
    bus                 -> bus
    truck, tricycle,
    awning-tricycle     -> truck

(VisDrone category 0 "ignored regions" and score==0 boxes are dropped.)

Usage:
    python prepare_visdrone_dataset.py --out data/visdrone_vn

Output:
    data/visdrone_vn/images/{train,val}/*.jpg
    data/visdrone_vn/labels/{train,val}/*.txt
    data/visdrone_vn.yaml   <- ultralytics dataset config

Train zip is ~1.4 GB, val zip ~70 MB. Zips are kept under
data/visdrone_vn/_zips/ so the script can resume/skip re-downloads.
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

URLS = {
    'train': 'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
    'val':   'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
}

RAW_DIRNAME = {
    'train': 'VisDrone2019-DET-train',
    'val':   'VisDrone2019-DET-val',
}

# VisDrone category (1-indexed in raw annotations) -> 0-indexed -> VN class id
# 0 person, 1 car, 2 motorcycle, 3 bus, 4 truck
VISDRONE_TO_VN = {
    0: 0,  # pedestrian -> person
    1: 0,  # people     -> person
    2: 2,  # bicycle    -> motorcycle
    3: 1,  # car        -> car
    4: 1,  # van        -> car
    5: 4,  # truck      -> truck
    6: 4,  # tricycle   -> truck
    7: 4,  # awning-tricycle -> truck
    8: 3,  # bus        -> bus
    9: 2,  # motor      -> motorcycle
}

VN_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  zip already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    print(f"    -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(zip_path: Path, dest_dir: Path, expected_dirname: str):
    if (dest_dir / expected_dirname).exists():
        print(f"  already extracted: {dest_dir / expected_dirname}")
        return
    print(f"  extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def convert_box(img_size, box):
    img_w, img_h = img_size
    x, y, w, h = box
    dw, dh = 1.0 / img_w, 1.0 / img_h
    return (x + w / 2) * dw, (y + h / 2) * dh, w * dw, h * dh


def convert_split(raw_dir: Path, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ann_dir = raw_dir / 'annotations'
    img_dir = raw_dir / 'images'

    n_images, n_boxes = 0, 0
    for ann_file in sorted(ann_dir.glob('*.txt')):
        img_file = img_dir / (ann_file.stem + '.jpg')
        if not img_file.exists():
            continue

        img_w, img_h = Image.open(img_file).size

        lines = []
        for row in ann_file.read_text().strip().splitlines():
            parts = row.split(',')
            if len(parts) < 6:
                continue
            x, y, w, h, score, category = (int(p) for p in parts[:6])
            if score == 0 or category == 0:
                continue  # ignored region
            vn_cls = VISDRONE_TO_VN.get(category - 1)
            if vn_cls is None:
                continue
            cx, cy, nw, nh = convert_box((img_w, img_h), (x, y, w, h))
            lines.append(f"{vn_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1

        if not lines:
            continue

        shutil.copy(img_file, out_images / img_file.name)
        (out_labels / (ann_file.stem + '.txt')).write_text('\n'.join(lines) + '\n')
        n_images += 1

    return n_images, n_boxes


def write_yaml(out_root: Path, yaml_path: Path):
    lines = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(VN_CLASS_NAMES)}",
        f"names: {VN_CLASS_NAMES}",
        "",
    ]
    yaml_path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='data/visdrone_vn',
                         help='output dataset directory')
    parser.add_argument('--yaml', default='data/visdrone_vn.yaml',
                         help='path to write the ultralytics dataset config')
    args = parser.parse_args()

    out_root = Path(args.out)
    zips_dir = out_root / '_zips'

    for split, url in URLS.items():
        print(f"[{split}]")
        zip_path = zips_dir / Path(url).name
        download(url, zip_path)
        extract(zip_path, zips_dir, RAW_DIRNAME[split])

        raw_dir = zips_dir / RAW_DIRNAME[split]
        n_images, n_boxes = convert_split(
            raw_dir,
            out_images=out_root / 'images' / split,
            out_labels=out_root / 'labels' / split,
        )
        print(f"  converted {n_images} images, {n_boxes} boxes -> "
              f"{out_root / 'images' / split}")

    write_yaml(out_root, Path(args.yaml))
    print(f"\nDataset config written to {args.yaml}")
    print("You can now run: python train_yolo_detector.py --data " + args.yaml)


if __name__ == '__main__':
    main()
