#!/usr/bin/env python3
"""
Ap danh sach sua ID (ghi tay sau khi xem review.mp4) vao 1 ban copy cua
pred_<CAM_ID>.txt, xuat ra gt_<CAM_ID>.txt — dung de tao ground-truth
"thay vi gan nhan tu dau" cho danh gia MOTA/IDF1 tren video CCTV thuc (xem
plan lay ground-truth ByteTrack).

File corrections (CSV, tu viet tay sau khi xem review.mp4):
    old_id,new_id,frame_start,frame_end
    1042,1023,60,110

Nghia: trong khoang frame [frame_start, frame_end], doi tat ca dong co
id == old_id thanh new_id (giu nguyen box/class). Cac dong sua duoc ap THEO
THU TU trong file — neu 1 doan ID-switch dai qua nhieu doi-ten lien tiep
(A->B->C), ghi 2 dong rieng theo dung thu tu, dong sau se thay doi id da
duoc dong truoc sua.

Usage:
    python scripts/eval/apply_id_corrections.py \
        --pred data/eval_real/pred_CAM_PHO_HUE_REAL.txt \
        --corrections data/eval_real/id_corrections.csv \
        --out data/eval_real/gt_CAM_PHO_HUE_REAL.txt
"""

import argparse
import csv
import sys


def load_corrections(path: str):
    rules = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules.append({
                'old_id': int(row['old_id']),
                'new_id': int(row['new_id']),
                'frame_start': int(row['frame_start']),
                'frame_end': int(row['frame_end']),
            })
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--corrections', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rules = load_corrections(args.corrections)
    print(f"Da doc {len(rules)} quy tac sua tu {args.corrections}")

    ids_before = set()
    ids_after = set()
    n_rows = 0
    n_changed = 0

    with open(args.pred, newline='', encoding='utf-8') as f_in, \
         open(args.out, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            if not row:
                continue
            n_rows += 1
            frame, gid, x1, y1, x2, y2, cls = row
            frame_i = int(frame)
            gid_i = int(gid)
            ids_before.add(gid_i)

            new_gid = gid_i
            for rule in rules:
                if new_gid == rule['old_id'] and rule['frame_start'] <= frame_i <= rule['frame_end']:
                    new_gid = rule['new_id']
            if new_gid != gid_i:
                n_changed += 1
            ids_after.add(new_gid)

            writer.writerow([frame, new_gid, x1, y1, x2, y2, cls])

    print(f"{n_rows} dong, {n_changed} dong bi doi id")
    print(f"So ID duy nhat: {len(ids_before)} (truoc) -> {len(ids_after)} (sau)")
    if len(ids_after) >= len(ids_before) and rules:
        print("CANH BAO: so ID khong giam sau khi sua — kiem tra lai "
              "file corrections (old_id/frame_start/frame_end co dung khong).")
    print(f"Da ghi {args.out}")


if __name__ == '__main__':
    main()
