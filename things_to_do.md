# Things To Do

## 1. QA ảnh Auto-Label (YOLO)

**Mục đích**: Kiểm tra chất lượng 15,957 ảnh đã được ITD model tự động gán nhãn trước khi dùng để fine-tune YOLO11s. Dữ liệu bẩn → model học pattern sai → accuracy kém.

**Vị trí dữ liệu**: `custom_tracking_system/data/auto_label/`  
**Preview grids**: 120 file ảnh lưới (mỗi file ~133 ảnh có bbox vẽ sẵn)

**Các bước cần làm**:
- [ ] Spot-check preview grids — nhìn nhanh bbox có bám vào xe/người đúng không
- [ ] Xác định ngưỡng lọc confidence (đề xuất: loại bỏ bbox < 0.4)
- [ ] Kiểm tra class imbalance (car vs motorcycle vs person)
- [ ] Loại bỏ ảnh không có object nào / bbox bị cắt quá nhỏ
- [ ] (Tùy chọn) Sửa tay ~200-500 ảnh sai rõ ràng bằng Label Studio / Roboflow

**Có thể bỏ qua sửa tay nếu**: preview grids trông hợp lý + đã filter theo confidence khi export.

---

## 2. Goal Classifier ✅ HOÀN THÀNH (baseline)

**Script**: `custom_tracking_system/scripts/train/train_goal_classifier.py`  
**Model**: `custom_tracking_system/weights/goal_classifier.pkl` (GradientBoosting, calibrated)  
**Data**: `custom_tracking_system/data/trajectories_hanoi_v2/` — 36,997 goals, moto 72.7%  
**Eval**: `custom_tracking_system/data/goal_classifier_eval_v3/`

**Kết quả đạt được (v3 — 26 features, window 4s):**

| Metric | Giá trị | Đánh giá |
|---|---|---|
| Accuracy @1.5s | 70.3% | Tốt cho prototype/dashboard |
| Accuracy @1.0s | 64.8% | Chấp nhận được (~7m trước ngã tư) |
| ECE (calibration) | 0.104 | Confidence tương đối đáng tin |
| Inference 30 xe | 4.0ms | Đạt yêu cầu CCTV thực tế |
| Motorcycle acc. | 70.2% | Đại diện cho traffic VN thực tế |

**Per-class recall:**
- straight: 75% ✅ — right: 86% ✅ — left: 53% ⚠️ — u_turn: 17% ❌

**Feature quan trọng nhất**: `head_sin`, `head_cos`, `speed_change_rate`, `head_rate_max`, `heading_range`

**Giới hạn đã xác định:**
- **U_turn recall 17%**: Xe quay đầu chỉ lộ signature khi đã vào ngã tư, không thể detect từ approach trajectory
- **Left recall 53%**: Còn yếu, cần cải thiện cho traffic HN
- **Sweet spot**: 1.0–1.5s trước ngã tư (~7–10m) là khoảng dự đoán tốt nhất

**Phù hợp dùng cho**: Dashboard hiển thị xác suất, thống kê lưu lượng theo hướng.  
**Chưa phù hợp cho**: Trigger cảnh báo tự động, điều khiển đèn.

**Để cải thiện tiếp** (khi cần, theo thứ tự ưu tiên):
1. Predict trong ngã tư (trong khi xe đang traversal) → u_turn recall kỳ vọng ~70%
2. LSTM trên raw sequence → +5–8% accuracy tổng thể
3. Thêm road topology feature từ OpenDRIVE (entry_lane_id, junction type)

**Re-train khi cần:**
```bat
python custom_tracking_system/scripts/train/train_goal_classifier.py ^
    --data custom_tracking_system/data/trajectories_hanoi_v2 ^
    --out-eval custom_tracking_system/data/goal_classifier_eval_v3
```

---

## 3. YOLO11s Fine-tune

**Phụ thuộc**: Hoàn thành QA ảnh (bước 1) trước  
**Yêu cầu**: GPU cloud (Kaggle / Colab) — không train được trên RX 6700XT (no CUDA Windows)

---

## 4. OSNet ReID Fine-tune

**Script**: `custom_tracking_system/train_veri.py`  
**Dataset**: `VeRi/` (đã có)  
**Yêu cầu**: GPU cloud (Kaggle / Colab)

---

## 5. Download thêm dataset cho YOLO11s

- VisDrone
- DriveIndia  
- UA-DETRAC
