#!/usr/bin/env python3
"""
Phân tích ngưỡng IncidentDetector dựa trên dữ liệu thật đã gán nhãn tay.

Đầu vào: thư mục snapshot do modules/incident_snapshot.py tạo ra (1 file
.jpg + 1 file .json mỗi incident), với trường "label" trong mỗi .json đã
được điền tay là "tp" (true positive) hoặc "fp" (false positive) — xem
docs/plan_tinh_chinh_nguong_incident.md, Bước 2.

Với mỗi loại luật (SUDDEN_STOP, SUDDEN_ACCEL, VEHICLE_PROXIMITY,
PREDICTED_COLLISION), script:
  1. Tách giá trị đã kích hoạt luật đó theo nhãn tp/fp.
  2. Vẽ histogram so sánh 2 nhóm, lưu PNG.
  3. Nếu đủ mẫu, tính precision/recall theo từng ngưỡng ứng viên và đề xuất
     ngưỡng mới (so với ngưỡng hiện tại trong config).

Usage:
    python scripts/eval/tune_incident_thresholds.py \
        --snapshot-dir docs/assets/incident_review \
        --out-dir docs/assets/incident_threshold_tuning
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Ngưỡng hiện tại (mặc định trong IncidentDetector.__init__, xem
# modules/incident_detector.py) — dùng làm mốc so sánh khi in báo cáo.
CURRENT_THRESHOLDS = {
    'SUDDEN_STOP': {'param': 'sudden_stop_ratio', 'value': 0.3},
    'SUDDEN_ACCEL': {'param': 'sudden_accel_ratio', 'value': 2.5},
    'VEHICLE_PROXIMITY': {'param': 'proximity_px', 'value': 100},
    'PREDICTED_COLLISION': {'param': 'ped_proximity_px * 1.5', 'value': 120},
}

# Với mỗi loại luật: tên trường trong "details" dùng làm giá trị kích hoạt,
# và "direction" cho biết giá trị CAO hơn có nghĩa "nghi ngờ hơn" (True) hay
# "an toàn hơn" (False) — cần để tính precision/recall đúng chiều.
RULE_VALUE_SPEC = {
    'SUDDEN_STOP': {
        'compute': lambda d: d['speed_after'] / max(d['speed_before'], 1e-6),
        'label': 'speed_after / speed_before (thấp hơn = nghi ngờ hơn)',
        'higher_is_suspicious': False,
    },
    'SUDDEN_ACCEL': {
        'compute': lambda d: d['speed_now'] / max(d['speed_prev'], 1e-6),
        'label': 'speed_now / speed_prev (cao hơn = nghi ngờ hơn)',
        'higher_is_suspicious': True,
    },
    'VEHICLE_PROXIMITY': {
        'compute': lambda d: d['distance_px'],
        'label': 'distance_px (thấp hơn = nghi ngờ hơn)',
        'higher_is_suspicious': False,
    },
    'PREDICTED_COLLISION': {
        'compute': lambda d: d['predicted_distance'],
        'label': 'predicted_distance (thấp hơn = nghi ngờ hơn)',
        'higher_is_suspicious': False,
    },
}


def load_labeled_incidents(snapshot_dir: Path) -> list[dict]:
    records = []
    for json_path in sorted(snapshot_dir.glob('*.json')):
        meta = json.loads(json_path.read_text(encoding='utf-8'))
        if meta.get('label') not in ('tp', 'fp'):
            continue  # chưa gán nhãn, bỏ qua
        records.append(meta)
    return records


def analyze_rule(rule_type: str, records: list[dict], out_dir: Path):
    spec = RULE_VALUE_SPEC.get(rule_type)
    if spec is None:
        print(f"[SKIP] {rule_type}: chưa khai báo cách tính giá trị kích hoạt "
              f"trong RULE_VALUE_SPEC")
        return

    tp_vals, fp_vals = [], []
    for r in records:
        if r['type'] != rule_type:
            continue
        try:
            value = spec['compute'](r['details'])
        except (KeyError, ZeroDivisionError):
            continue
        (tp_vals if r['label'] == 'tp' else fp_vals).append(value)

    n_tp, n_fp = len(tp_vals), len(fp_vals)
    print(f"\n=== {rule_type} === (n_tp={n_tp}, n_fp={n_fp})")
    if n_tp == 0 or n_fp == 0:
        print("  Chưa đủ mẫu cả 2 nhãn — cần gán nhãn thêm trước khi phân tích.")
        return

    tp_arr, fp_arr = np.array(tp_vals), np.array(fp_vals)
    print(f"  TP {spec['label']}: mean={tp_arr.mean():.2f} std={tp_arr.std():.2f} "
          f"min={tp_arr.min():.2f} max={tp_arr.max():.2f}")
    print(f"  FP {spec['label']}: mean={fp_arr.mean():.2f} std={fp_arr.std():.2f} "
          f"min={fp_arr.min():.2f} max={fp_arr.max():.2f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fp_arr, bins=15, alpha=0.6, label=f'FP (n={n_fp})', color='tab:red')
        ax.hist(tp_arr, bins=15, alpha=0.6, label=f'TP (n={n_tp})', color='tab:green')
        ax.set_xlabel(spec['label'])
        ax.set_ylabel('Số lượng')
        ax.set_title(f'{rule_type}: phân phối giá trị kích hoạt theo nhãn')
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f'{rule_type.lower()}_tp_vs_fp.png'
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Đã lưu histogram -> {out_path}")
    except ImportError:
        print("  (matplotlib không có sẵn — bỏ qua bước vẽ histogram)")

    if n_tp < 20 or n_fp < 20:
        print(f"  Số mẫu còn ít (khuyến nghị >= 20 mỗi nhóm) — ngưỡng đề "
              f"xuất dưới đây chỉ mang tính tham khảo, không nên áp dụng "
              f"ngay nếu chưa xem qua histogram.")

    # Quét ngưỡng ứng viên, tính precision/recall coi "nghi ngờ" là positive.
    labels = np.array([1] * n_tp + [0] * n_fp)
    values = np.concatenate([tp_arr, fp_arr])
    candidates = np.unique(values)
    best = None
    for thr in candidates:
        if spec['higher_is_suspicious']:
            pred_positive = values >= thr
        else:
            pred_positive = values <= thr
        tp = int(np.sum(pred_positive & (labels == 1)))
        fp = int(np.sum(pred_positive & (labels == 0)))
        fn = int(np.sum(~pred_positive & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if best is None or f1 > best['f1']:
            best = {'threshold': float(thr), 'precision': precision,
                     'recall': recall, 'f1': f1}

    cur = CURRENT_THRESHOLDS.get(rule_type)
    print(f"  Ngưỡng hiện tại: {cur['param']} = {cur['value']}")
    print(f"  Ngưỡng đề xuất (tối ưu F1 trên tập đã gán nhãn, IN-SAMPLE — "
          f"xem Bước 5 để validate độc lập):")
    print(f"    threshold={best['threshold']:.3f}  "
          f"precision={best['precision']:.2f}  recall={best['recall']:.2f}  "
          f"f1={best['f1']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=str,
                         default='docs/assets/incident_review')
    parser.add_argument('--out-dir', type=str,
                         default='docs/assets/incident_threshold_tuning')
    args = parser.parse_args()

    snapshot_dir = project_root / args.snapshot_dir
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not snapshot_dir.exists():
        print(f"Không tìm thấy {snapshot_dir} — chạy Bước 1 (main.py "
              f"--incident-snapshot-dir ...) và Bước 2 (gán nhãn tay) trước.")
        sys.exit(1)

    records = load_labeled_incidents(snapshot_dir)
    print(f"Đã nạp {len(records)} incident có nhãn (tp/fp) từ {snapshot_dir}")
    if not records:
        print("Chưa có incident nào được gán nhãn — điền trường \"label\" "
              "trong các file .json (\"tp\" hoặc \"fp\") rồi chạy lại.")
        sys.exit(1)

    for rule_type in RULE_VALUE_SPEC:
        analyze_rule(rule_type, records, out_dir)


if __name__ == '__main__':
    main()
