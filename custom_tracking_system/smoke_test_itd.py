"""
ITD Model Smoke Test — đánh giá transfer sang camera giao thông Hà Nội.

So sánh:
  - best_xl_ITD_v1.2.pt  (yolo11x, 8 class Ấn Độ) — remap → 5 class
  - weights/yolo11m_vn.pt (yolo11m, carla6 checkpoint)

Output:
  - Terminal: stats per video, per model
  - smoke_test_output/<video>_itd.jpg   — annotated frame ITD
  - smoke_test_output/<video>_vn.jpg    — annotated frame VN model
  - smoke_test_output/report.json       — full numeric report
"""

import os, sys, json, time, textwrap
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_ITD = BASE / "weights" / "best_xl_ITD_v1.2.pt"
MODEL_VN  = BASE / "weights" / "yolo11m_vn.pt"
OUT_DIR   = BASE / "smoke_test_output"
OUT_DIR.mkdir(exist_ok=True)

DATA_ROOT = BASE / "data"
D02 = DATA_ROOT / "datasets" / "traffic_intersection" / "02. Dữ liệu camera giao thông"
D03 = DATA_ROOT / "datasets" / "traffic_intersection_multi" / "03. Dữ liệu camera giao thông các ngã tư gần nhau cùng thời điểm"

TEST_VIDEOS = [
    {
        "name": "noi_do_HN_sang",
        "path": D02 / "Giao thông nội đô HN" / "20221003-102452.mp4",
        "desc": "Nội đô HN — sáng 10:24",
    },
    {
        "name": "nga_tu_Xa_Dan_trua",
        "path": D02 / "Ngã tư Xã Đàn Phạm Ngọc Thạch" / "12.10.9.22.mp4",
        "desc": "Ngã tư Xã Đàn – Phạm Ngọc Thạch — trưa 12h",
    },
    {
        "name": "Ly_Thuong_Kiet_Ba_Trieu_16h",
        "path": D03 / "Lý Thường Kiệt - Bà Triệu" / "16h.15.9.22.mp4",
        "desc": "Lý Thường Kiệt – Bà Triệu — 16h cao điểm",
    },
]

SAMPLE_FRAMES = 120   # số frame lấy mẫu mỗi video (phân bố đều)
CONF          = 0.25  # ngưỡng confidence (hạ thấp để quan sát cả detection yếu)
IMG_SIZE      = 640

# ---------------------------------------------------------------------------
# ITD class remap → 5 class chung
# 8 class gốc: two wheeler, autorickshaw, car, bus, LCV, truck, bicycle, pedestrian
# ---------------------------------------------------------------------------
ITD_REMAP = {
    "two wheeler":  "motorcycle",
    "two_wheeler":  "motorcycle",
    "autorickshaw": "motorcycle",
    "car":          "car",
    "bus":          "bus",
    "lcv":          "truck",
    "LCV":          "truck",
    "truck":        "truck",
    "bicycle":      None,        # bỏ
    "pedestrian":   "person",
    "pedestrain":   "person",   # typo trong ITD model weights
}

VN_CLASSES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}

COLORS = {
    "person":     (255,  60,  60),
    "car":        ( 60, 220,  60),
    "motorcycle": ( 60, 220, 220),
    "bus":        ( 60,  60, 255),
    "truck":      (220, 220,  60),
    None:         (160, 160, 160),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path):
    from ultralytics import YOLO
    m = YOLO(str(path))
    return m


def sample_video(path, n=SAMPLE_FRAMES):
    """Trả về list frame (BGR np.array) lấy đều từ video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = np.linspace(0, max(total - 1, 0), min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames, fps, w, h, total


def remap_itd_class(raw_name: str) -> str | None:
    key = raw_name.strip().lower().replace(" ", "_")
    # thử key gốc lowercase
    for k, v in ITD_REMAP.items():
        if k.lower().replace(" ", "_") == key:
            return v
    return None   # không nhận ra → bỏ


def run_model(model, frames, class_map_fn):
    """
    class_map_fn(raw_name) -> canonical_name | None
    Trả về list of list of dict: [{cls, conf, box}, ...]
    """
    all_dets = []
    t0 = time.perf_counter()
    for frame in frames:
        results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        frame_dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names
            for box in boxes:
                raw  = names[int(box.cls[0])]
                cls  = class_map_fn(raw)
                if cls is None:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                frame_dets.append({"cls": cls, "conf": conf, "box": xyxy.tolist()})
        all_dets.append(frame_dets)
    elapsed = time.perf_counter() - t0
    return all_dets, elapsed


def compute_stats(all_dets, n_frames):
    cls_count   = defaultdict(int)
    cls_conf    = defaultdict(list)
    empty_frames = 0
    total_dets   = 0

    for frame_dets in all_dets:
        if not frame_dets:
            empty_frames += 1
        for d in frame_dets:
            cls_count[d["cls"]] += 1
            cls_conf[d["cls"]].append(d["conf"])
            total_dets += 1

    stats = {
        "total_frames":   n_frames,
        "empty_frames":   empty_frames,
        "empty_pct":      round(100 * empty_frames / max(n_frames, 1), 1),
        "total_dets":     total_dets,
        "avg_dets_frame": round(total_dets / max(n_frames, 1), 2),
        "per_class": {},
    }
    for cls in sorted(cls_count):
        confs = cls_conf[cls]
        stats["per_class"][cls] = {
            "count":    cls_count[cls],
            "avg_conf": round(np.mean(confs), 3),
            "min_conf": round(np.min(confs), 3),
            "max_conf": round(np.max(confs), 3),
        }
    return stats


def draw_detections(frame, dets, label_prefix=""):
    out = frame.copy()
    for d in dets:
        cls  = d["cls"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return out


def side_by_side(frame, dets_itd, dets_vn, video_name, frame_idx):
    h, w = frame.shape[:2]
    left  = draw_detections(frame, dets_itd, "ITD:")
    right = draw_detections(frame, dets_vn,  "VN:")

    # Header
    def add_header(img, title, n_det):
        bar = np.zeros((30, w, 3), dtype=np.uint8)
        cv2.putText(bar, f"{title}  |  {n_det} det", (6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return np.vstack([bar, img])

    left  = add_header(left,  "ITD (yolo11x remap)", len(dets_itd))
    right = add_header(right, "VN  (yolo11m_vn)",    len(dets_vn))

    canvas = np.hstack([left, right])

    # Watermark
    cv2.putText(canvas, f"{video_name}  frame={frame_idx}", (8, h + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def print_stats(label, stats, elapsed, n_frames):
    fps_inf = n_frames / max(elapsed, 1e-6)
    lines = [
        f"\n{'='*60}",
        f"  {label}",
        f"{'='*60}",
        f"  Frames sampled : {stats['total_frames']}",
        f"  Empty frames   : {stats['empty_frames']}  ({stats['empty_pct']}%)",
        f"  Total dets     : {stats['total_dets']}",
        f"  Avg dets/frame : {stats['avg_dets_frame']}",
        f"  Inference time : {elapsed:.1f}s  (~{fps_inf:.1f} FPS)",
        f"  Per-class breakdown:",
    ]
    for cls, info in stats["per_class"].items():
        lines.append(
            f"    {cls:<12} count={info['count']:>5}  "
            f"conf avg={info['avg_conf']:.3f}  "
            f"[{info['min_conf']:.2f}–{info['max_conf']:.2f}]"
        )
    if not stats["per_class"]:
        lines.append("    (no detections)")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== ITD Smoke Test ===")
    print(f"ITD model : {MODEL_ITD}")
    print(f"VN  model : {MODEL_VN}")
    print(f"Output    : {OUT_DIR}")
    print(f"Frames/video: {SAMPLE_FRAMES}  conf>={CONF}\n")

    if not MODEL_ITD.exists():
        sys.exit(f"[ERROR] Không tìm thấy ITD model: {MODEL_ITD}")
    if not MODEL_VN.exists():
        sys.exit(f"[ERROR] Không tìm thấy VN model: {MODEL_VN}")

    print("Đang load models...")
    itd_model = load_model(MODEL_ITD)
    vn_model  = load_model(MODEL_VN)

    # In class names của ITD để verify
    print(f"\n[ITD raw classes] {itd_model.names}")
    print(f"[VN  raw classes] {vn_model.names}\n")

    # Xác định class_map_fn cho VN model (CLASSES_VN hoặc COCO)
    vn_names = vn_model.names  # {id: name}
    def vn_map(raw_name):
        # raw_name đã là tên class → trả thẳng nếu trong 5 class
        if raw_name in ("person", "car", "motorcycle", "bus", "truck"):
            return raw_name
        # fallback: COCO mapping
        coco = {"person": "person", "car": "car", "motorcycle": "motorcycle",
                "bus": "bus", "truck": "truck"}
        return coco.get(raw_name, None)

    full_report = {}

    for video_info in TEST_VIDEOS:
        vpath = video_info["path"]
        vname = video_info["name"]
        desc  = video_info["desc"]

        if not vpath.exists():
            print(f"[SKIP] Không tìm thấy: {vpath}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Video: {desc}")
        print(f"  Path : {vpath.name}")

        frames, fps, w, h, total = sample_video(vpath, SAMPLE_FRAMES)
        n = len(frames)
        print(f"  Độ phân giải: {w}×{h}  FPS: {fps:.1f}  Tổng frame: {total}  → Lấy mẫu: {n}")

        # --- Chạy ITD ---
        print("  [ITD] Đang chạy inference...")
        dets_itd_all, t_itd = run_model(itd_model, frames, remap_itd_class)
        stats_itd = compute_stats(dets_itd_all, n)
        print_stats(f"ITD  |  {desc}", stats_itd, t_itd, n)

        # --- Chạy VN ---
        print("  [VN] Đang chạy inference...")
        dets_vn_all, t_vn = run_model(vn_model, frames, vn_map)
        stats_vn = compute_stats(dets_vn_all, n)
        print_stats(f"VN   |  {desc}", stats_vn, t_vn, n)

        # --- Lưu ảnh annotated (frame giữa video) ---
        mid = n // 2
        frame_mid = frames[mid]

        # Ảnh riêng
        cv2.imwrite(str(OUT_DIR / f"{vname}_itd.jpg"),
                    draw_detections(frame_mid, dets_itd_all[mid], ""))
        cv2.imwrite(str(OUT_DIR / f"{vname}_vn.jpg"),
                    draw_detections(frame_mid, dets_vn_all[mid],  ""))

        # Ảnh so sánh side-by-side
        cmp = side_by_side(frame_mid, dets_itd_all[mid], dets_vn_all[mid], vname, mid)
        cv2.imwrite(str(OUT_DIR / f"{vname}_compare.jpg"), cmp)

        # Lưu thêm frame đầu + frame cuối để quan sát cảnh
        for tag, fi in [("first", 0), ("last", n - 1)]:
            cmp2 = side_by_side(frames[fi], dets_itd_all[fi], dets_vn_all[fi], vname, fi)
            cv2.imwrite(str(OUT_DIR / f"{vname}_{tag}_compare.jpg"), cmp2)

        full_report[vname] = {
            "desc":    desc,
            "video":   str(vpath.name),
            "resolution": f"{w}x{h}",
            "fps":     fps,
            "total_frames": total,
            "sampled": n,
            "itd": {**stats_itd, "elapsed_s": round(t_itd, 2)},
            "vn":  {**stats_vn,  "elapsed_s": round(t_vn, 2)},
        }

    # --- Lưu report JSON ---
    report_path = OUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # --- Summary cuối ---
    print(f"\n{'='*60}")
    print("  TỔNG KẾT SMOKE TEST")
    print(f"{'='*60}")
    print(f"  {'Video':<30} {'Metric':<25} {'ITD':>8} {'VN':>8}")
    print(f"  {'-'*30} {'-'*25} {'-'*8} {'-'*8}")
    for vname, rep in full_report.items():
        si = rep["itd"]
        sv = rep["vn"]
        vdesc = rep["desc"][:28]
        rows = [
            ("% frames trống",  si["empty_pct"],      sv["empty_pct"]),
            ("avg dets/frame",  si["avg_dets_frame"],  sv["avg_dets_frame"]),
            ("motorcycle count",
             si["per_class"].get("motorcycle", {}).get("count", 0),
             sv["per_class"].get("motorcycle", {}).get("count", 0)),
            ("car count",
             si["per_class"].get("car", {}).get("count", 0),
             sv["per_class"].get("car", {}).get("count", 0)),
        ]
        first = True
        for metric, vi, vv in rows:
            lbl = vdesc if first else ""
            print(f"  {lbl:<30} {metric:<25} {str(vi):>8} {str(vv):>8}")
            first = False
        print()

    print(f"  Ảnh annotated  : {OUT_DIR}/<video>_compare.jpg")
    print(f"  Report JSON    : {report_path}")
    print(f"\n  Diễn giải nhanh:")
    print(textwrap.dedent("""
      - % frames trống cao  → model bỏ sót nhiều, domain gap lớn
      - motorcycle count    → class quan trọng nhất với traffic VN
      - conf avg thấp (<0.4) → detection không chắc chắn, nhiều FP/FN
      - So ITD vs VN:
          ITD tốt hơn  → ITD dùng được làm auto-labeler cho data VN
          ITD tệ hơn  → ITD gap quá lớn, chỉ dùng CARLA render + VisDrone
    """))


if __name__ == "__main__":
    main()
