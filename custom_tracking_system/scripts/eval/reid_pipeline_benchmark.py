#!/usr/bin/env python3
"""
Benchmark OSNet (DualReIDExtractor) tren crop THAT, nhieu, lay truc tiep tu
video CCTV — khong dung crop sach cua VeRi-776 (Bang 4.4 chi do retrieval
ngoai tuyen tren crop sach, chua phan anh do chinh xac khi dau vao la crop
nhieu tu detection/tracking truc tiep — xem Chuong 6 muc 6.1).

Khong co video 2-camera lien tuyen (UC2) nen KHONG do duoc "ReID xuyen
camera" dung nghia — day la mot proxy trong pipeline 1 camera: cap "cung 1
xe" duoc lay tu chinh id_corrections.csv (nguoi xem da xac nhan bang mat la
cung 1 xe nhung ByteTrack gan 2 id khac nhau — dung 2 crop o 2 thoi diem
khac nhau, dung dinh nghia "anh nhieu, that, kho" can de kiem tra ReID), cap
"khac xe" lay tu 2 global_id khac nhau bat ky (da qua sua tay, dang tin la
dung). Bao cao do tach biet similarity giua 2 nhom — khong phai con so
"ReID xuyen camera" cuoi cung.

Usage:
    python scripts/eval/reid_pipeline_benchmark.py \
        --video <path.mp4> \
        --pred data/eval_real/pred_CAM_PHO_HUE_REAL.txt \
        --corrections data/eval_real/id_corrections.csv \
        --n-negative 30
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.reid import DualReIDExtractor


def load_pred(path):
    by_frame_id = defaultdict(dict)   # frame -> {id: (box, cls)}
    by_id_frames = defaultdict(list)  # id -> [frame, ...] sorted
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, gid, x1, y1, x2, y2, cls = row
            frame, gid = int(frame), int(gid)
            box = (float(x1), float(y1), float(x2), float(y2))
            by_frame_id[frame][gid] = (box, cls)
            by_id_frames[gid].append(frame)
    for gid in by_id_frames:
        by_id_frames[gid].sort()
    return by_frame_id, by_id_frames


def load_corrections(path):
    rules = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules.append({
                'old_id': int(row['old_id']), 'new_id': int(row['new_id']),
                'frame_start': int(row['frame_start']), 'frame_end': int(row['frame_end']),
            })
    return rules


def get_frame(cap, cache, frame_idx):
    if frame_idx not in cache:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame_bgr = cap.read()
        cache[frame_idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if ret else None
    return cache[frame_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--corrections', required=True)
    ap.add_argument('--n-negative', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    by_frame_id, by_id_frames = load_pred(args.pred)
    rules = load_corrections(args.corrections)
    cap = cv2.VideoCapture(args.video)
    frame_cache = {}

    extractor = DualReIDExtractor()

    def crop_feature(frame_idx, gid):
        info = by_frame_id.get(frame_idx, {}).get(gid)
        if info is None:
            return None, None
        box, cls = info
        frame = get_frame(cap, frame_cache, frame_idx)
        if frame is None:
            return None, None
        feat = extractor.extract_feature(frame, [int(v) for v in box], object_class=cls)
        return feat, cls

    # --- Positive pairs: tu id_corrections.csv (nguoi xem xac nhan cung 1 xe) ---
    positives = []
    for rule in rules:
        new_frames_before = [f for f in by_id_frames.get(rule['new_id'], [])
                              if f < rule['frame_start']]
        old_frames_in_range = [f for f in by_id_frames.get(rule['old_id'], [])
                                if rule['frame_start'] <= f <= rule['frame_end']]
        if not new_frames_before or not old_frames_in_range:
            continue
        f1 = new_frames_before[-1]                       # ngay truoc luc switch
        f2 = old_frames_in_range[len(old_frames_in_range) // 2]  # giua doan switch
        positives.append((rule['new_id'], f1, rule['old_id'], f2))

    # --- Negative pairs: 2 global_id khac nhau bat ky, cung class, ngau nhien ---
    all_ids = list(by_id_frames.keys())
    negatives = []
    attempts = 0
    while len(negatives) < args.n_negative and attempts < args.n_negative * 20:
        attempts += 1
        gid_a, gid_b = random.sample(all_ids, 2)
        fa = random.choice(by_id_frames[gid_a])
        fb = random.choice(by_id_frames[gid_b])
        cls_a = by_frame_id[fa][gid_a][1]
        cls_b = by_frame_id[fb][gid_b][1]
        if cls_a != cls_b:
            continue
        negatives.append((gid_a, fa, gid_b, fb))

    print(f"{len(positives)} cap CUNG xe (tu id_corrections.csv), "
          f"{len(negatives)} cap KHAC xe (ngau nhien)\n")

    def evaluate(pairs, label):
        sims = []
        for gid_a, fa, gid_b, fb in pairs:
            feat_a, cls_a = crop_feature(fa, gid_a)
            feat_b, cls_b = crop_feature(fb, gid_b)
            if feat_a is None or feat_b is None:
                continue
            sim = float(np.dot(feat_a.ravel(), feat_b.ravel()))
            sims.append(sim)
            print(f"  [{label}] id {gid_a}@f{fa} vs id {gid_b}@f{fb} "
                  f"({cls_a}) sim={sim:.3f}")
        return sims

    pos_sims = evaluate(positives, "CUNG")
    neg_sims = evaluate(negatives, "KHAC")

    print(f"\n=== Ket qua ===")
    print(f"Cung xe   : n={len(pos_sims)}  mean={np.mean(pos_sims):.3f}  "
          f"min={np.min(pos_sims):.3f}  max={np.max(pos_sims):.3f}")
    print(f"Khac xe   : n={len(neg_sims)}  mean={np.mean(neg_sims):.3f}  "
          f"min={np.min(neg_sims):.3f}  max={np.max(neg_sims):.3f}")

    threshold = 0.5  # khop nguong thuc te cua GlobalTracker (modules/global_tracking.py)
    tp = sum(1 for s in pos_sims if s >= threshold)
    fn = len(pos_sims) - tp
    tn = sum(1 for s in neg_sims if s < threshold)
    fp = len(neg_sims) - tn
    print(f"\nTai nguong san xuat threshold={threshold}:")
    print(f"  Cung xe nhung KHONG ghep duoc (FN, mat track that): {fn}/{len(pos_sims)}")
    print(f"  Khac xe nhung BI ghep nham (FP, nhap nhan 2 xe): {fp}/{len(neg_sims)}")
    print(f"  Accuracy: {(tp + tn) / max(1, len(pos_sims) + len(neg_sims)):.1%}")


if __name__ == '__main__':
    main()
