"""
Auto-label toàn bộ video Hanoi traffic bằng ITD model (yolo11x).

Output (YOLO format):
  auto_label_output/
    images/         <- frame .jpg
    labels/         <- YOLO .txt (cx cy w h normalized, 1 line/bbox)
    preview/        <- annotated .jpg cho QA tay
    dataset.yaml    <- ultralytics dataset config
    label_stats.json

Class IDs (5-class project):
  0: person | 1: car | 2: motorcycle | 3: bus | 4: truck

Usage:
  python auto_label_itd.py
  python auto_label_itd.py --max-frames 200 --conf 0.35 --batch 4
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_ROOT = BASE / "data"
MODEL_PATH = BASE / "weights" / "best_xl_ITD_v1.2.pt"
OUT_DIR   = BASE / "data" / "auto_label"

D02 = DATA_ROOT / "datasets" / "traffic_intersection" / "02. Dữ liệu camera giao thông"
D03 = DATA_ROOT / "datasets" / "traffic_intersection_multi" / "03. Dữ liệu camera giao thông các ngã tư gần nhau cùng thời điểm"
SEARCH_ROOTS = [D02, D03]
VIDEO_EXTS   = {".mp4", ".avi", ".ave", ".mov", ".mkv"}

# ──────────────────────────────────────────────────────────
# ITD -> 5-class remap
# ITD raw: two wheeler, autorickshaw, car, bus, LCV, truck, bicycle, pedestrain
# ──────────────────────────────────────────────────────────
YOLO_CLASSES = {
    "person":     0,
    "car":        1,
    "motorcycle": 2,
    "bus":        3,
    "truck":      4,
}

ITD_REMAP = {
    "two wheeler":  "motorcycle",
    "two_wheeler":  "motorcycle",
    "autorickshaw": "motorcycle",
    "car":          "car",
    "bus":          "bus",
    "lcv":          "truck",
    "truck":        "truck",
    "bicycle":      None,
    "pedestrian":   "person",
    "pedestrain":   "person",   # typo trong model weights
}

CLASS_COLORS = {
    0: (255,  60,  60),   # person  — đỏ
    1: ( 60, 220,  60),   # car     — xanh lá
    2: ( 60, 220, 220),   # motorcycle — cyan
    3: ( 60,  60, 255),   # bus     — xanh dương
    4: (220, 220,  60),   # truck   — vàng
}
CLASS_NAMES = {v: k for k, v in YOLO_CLASSES.items()}


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.replace(" ", "_").replace("(", "").replace(")", "")
    text = re.sub(r"[^\w\-]", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:50]


def find_videos(roots) -> list[dict]:
    videos = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                # source_tag = subfolder + filename
                rel_parts = p.relative_to(root).parts
                tag = slugify("_".join(rel_parts[:-1])) + "__" + slugify(p.stem)
                tag = tag[:70]
                videos.append({"path": p, "tag": tag})
    return videos


def remap_itd(raw_name: str):
    key = raw_name.strip().lower().replace(" ", "_")
    for k, v in ITD_REMAP.items():
        if k.lower().replace(" ", "_") == key:
            return v
    return None


def sample_indices(total_frames: int, fps: float, max_frames: int) -> list[int]:
    """Lấy 1 frame mỗi 0.5 giây, tối đa max_frames."""
    step = max(1, int(fps * 0.5))
    indices = list(range(0, total_frames, step))
    if len(indices) > max_frames:
        indices = [indices[int(i * (len(indices) - 1) / (max_frames - 1))]
                   for i in range(max_frames)]
    return sorted(set(indices))


def draw_labels(frame, dets):
    out = frame.copy()
    for cls_id, conf, x1, y1, x2, y2 in dets:
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return out


def make_preview_grid(frames_dets: list, video_tag: str, n_cols: int = 3) -> np.ndarray:
    """Tạo grid ảnh annotated để preview."""
    cells = []
    for frame, dets in frames_dets:
        ann = draw_labels(frame, dets)
        # Resize về chiều cao cố định 300px
        h, w = ann.shape[:2]
        new_h = 300
        new_w = int(w * new_h / h)
        cells.append(cv2.resize(ann, (new_w, new_h)))

    if not cells:
        return np.zeros((300, 400, 3), dtype=np.uint8)

    # Pad tất cả về cùng width
    max_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        h, w = c.shape[:2]
        pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
        padded.append(np.hstack([c, pad]))

    # Xếp thành rows
    rows = []
    for i in range(0, len(padded), n_cols):
        row_cells = padded[i:i + n_cols]
        while len(row_cells) < n_cols:
            row_cells.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row_cells))
    grid = np.vstack(rows)

    # Watermark
    cv2.putText(grid, video_tag[:80], (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return grid


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Max frames sampled per video (default 150)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default 0.35)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Inference batch size (default 1)")
    parser.add_argument("--preview-per-video", type=int, default=6,
                        help="Annotated preview frames per video (default 6)")
    parser.add_argument("--skip-empty", action="store_true", default=True,
                        help="Bỏ qua frame không có detection (default True)")
    args = parser.parse_args()

    # Setup output dirs
    img_dir     = OUT_DIR / "images"
    lbl_dir     = OUT_DIR / "labels"
    preview_dir = OUT_DIR / "preview"
    for d in [img_dir, lbl_dir, preview_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load model
    from ultralytics import YOLO
    print(f"Loading ITD model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model_names = model.names
    print(f"  Classes: {model_names}\n")

    # Find all videos
    videos = find_videos(SEARCH_ROOTS)
    print(f"Found {len(videos)} video files\n")

    global_stats  = defaultdict(int)
    per_video     = {}
    total_t0      = time.perf_counter()
    frame_counter = 0  # global counter cho file naming

    for vi, vinfo in enumerate(videos):
        vpath = vinfo["path"]
        vtag  = vinfo["tag"]

        # Open video
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[{vi+1}/{len(videos)}] SKIP (cannot open): {vpath.name}")
            per_video[vtag] = {"status": "skip", "reason": "cannot_open"}
            continue

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 6.0
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_f <= 0:
            cap.release()
            print(f"[{vi+1}/{len(videos)}] SKIP (0 frames): {vpath.name}")
            per_video[vtag] = {"status": "skip", "reason": "zero_frames"}
            continue

        sample_idx = sample_indices(total_f, fps, args.max_frames)
        n_sample   = len(sample_idx)

        print(f"[{vi+1}/{len(videos)}] {vpath.parent.name}/{vpath.name}")
        print(f"  {w}x{h} {fps:.0f}fps  total={total_f}  sample={n_sample}")

        # Collect frames
        frames_map = {}   # frame_idx -> np.array
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_map[idx] = frame
        cap.release()

        if not frames_map:
            print("  -> No readable frames, skip")
            per_video[vtag] = {"status": "skip", "reason": "no_readable_frames"}
            continue

        # Inference in batches
        frame_keys = sorted(frames_map.keys())
        batch_size = args.batch
        all_results = {}   # frame_idx -> [(cls_id, conf, x1,y1,x2,y2)]

        t0_inf = time.perf_counter()
        for bi in range(0, len(frame_keys), batch_size):
            batch_keys   = frame_keys[bi:bi + batch_size]
            batch_frames = [frames_map[k] for k in batch_keys]
            results = model(batch_frames, imgsz=640, conf=args.conf, verbose=False)
            for key, r in zip(batch_keys, results):
                dets = []
                if r.boxes is not None:
                    for box in r.boxes:
                        raw_name = model_names[int(box.cls[0])]
                        mapped   = remap_itd(raw_name)
                        if mapped is None:
                            continue
                        cls_id = YOLO_CLASSES[mapped]
                        conf   = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        dets.append((cls_id, conf, x1, y1, x2, y2))
                all_results[key] = dets

        elapsed_inf = time.perf_counter() - t0_inf
        inf_fps     = len(frame_keys) / max(elapsed_inf, 1e-6)

        # Lưu images + labels, thu thập preview
        saved = 0
        skipped_empty = 0
        cls_counts = defaultdict(int)
        preview_frames_dets = []

        # Chọn indices cho preview (phân bố đều)
        n_prev = args.preview_per_video
        prev_indices = set()
        if n_prev > 0 and frame_keys:
            prev_step = max(1, len(frame_keys) // n_prev)
            prev_indices = {frame_keys[min(i * prev_step, len(frame_keys)-1)]
                            for i in range(n_prev)}

        for fidx in frame_keys:
            frame = frames_map[fidx]
            dets  = all_results[fidx]
            fh, fw = frame.shape[:2]

            if args.skip_empty and not dets:
                skipped_empty += 1
                if fidx in prev_indices:
                    preview_frames_dets.append((frame, dets))
                continue

            # File name
            stem = f"{vtag}__{frame_counter:07d}"
            frame_counter += 1

            # Save image
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save YOLO label
            lines = []
            for cls_id, conf, x1, y1, x2, y2 in dets:
                cx = ((x1 + x2) / 2) / fw
                cy = ((y1 + y2) / 2) / fh
                bw = (x2 - x1) / fw
                bh = (y2 - y1) / fh
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                cls_counts[cls_id] += 1
                global_stats[cls_id] += 1

            with open(lbl_dir / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines))

            saved += 1

            # Collect preview
            if fidx in prev_indices:
                preview_frames_dets.append((frame, dets))

        # Save preview grid
        if preview_frames_dets:
            grid = make_preview_grid(preview_frames_dets, vtag)
            cv2.imwrite(str(preview_dir / f"{vtag}_preview.jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

        cls_str = "  ".join(
            f"{CLASS_NAMES[k]}={v}" for k, v in sorted(cls_counts.items())
        )
        print(f"  saved={saved}  skip_empty={skipped_empty}  inf={elapsed_inf:.1f}s ({inf_fps:.1f}fps)")
        print(f"  dets: {cls_str or '(none)'}")

        per_video[vtag] = {
            "status":       "ok",
            "file":         str(vpath.name),
            "resolution":   f"{w}x{h}",
            "fps":          fps,
            "total_frames": total_f,
            "sampled":      n_sample,
            "saved":        saved,
            "skip_empty":   skipped_empty,
            "inf_fps":      round(inf_fps, 1),
            "class_counts": {CLASS_NAMES[k]: v for k, v in cls_counts.items()},
        }

    # ── Lưu dataset.yaml ──
    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR.resolve()}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: {len(YOLO_CLASSES)}\n")
        names_list = [CLASS_NAMES[i] for i in range(len(YOLO_CLASSES))]
        f.write(f"names: {names_list}\n")

    # ── Lưu stats JSON ──
    total_elapsed = time.perf_counter() - total_t0
    report = {
        "total_videos":  len(videos),
        "total_images":  frame_counter,
        "elapsed_min":   round(total_elapsed / 60, 1),
        "global_class_counts": {CLASS_NAMES[k]: v for k, v in sorted(global_stats.items())},
        "per_video": per_video,
    }
    with open(OUT_DIR / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    ok_videos  = sum(1 for v in per_video.values() if v.get("status") == "ok")
    skp_videos = len(videos) - ok_videos
    total_time = time.strftime("%Hh%Mm%Ss", time.gmtime(int(total_elapsed)))

    print("\n" + "="*60)
    print("  AUTO-LABEL DONE")
    print("="*60)
    print(f"  Videos OK / total : {ok_videos} / {len(videos)}  (skip={skp_videos})")
    print(f"  Total images saved: {frame_counter}")
    print(f"  Total time        : {total_time}")
    print(f"  Global class dist :")
    total_boxes = sum(global_stats.values())
    for cls_id in sorted(global_stats):
        n   = global_stats[cls_id]
        pct = 100 * n / max(total_boxes, 1)
        print(f"    {CLASS_NAMES[cls_id]:<12} {n:>7}  ({pct:.1f}%)")
    print(f"\n  Output : {OUT_DIR}")
    print(f"    images/  : {frame_counter} .jpg")
    print(f"    labels/  : {frame_counter} .txt  (YOLO format)")
    print(f"    preview/ : {ok_videos} _preview.jpg  (cho QA tay)")
    print(f"    dataset.yaml + label_stats.json")
    print("\n  Buoc tiep theo:")
    print("    1. QA thu cong: mo preview/*.jpg, kiem tra nhãn sai")
    print("    2. Dung LabelImg/Roboflow de sua nhan sai class")
    print("    3. Split train/val (80/20), them vao pipeline fine-tune YOLO11s")


if __name__ == "__main__":
    main()
