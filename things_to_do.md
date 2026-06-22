# Things To Do

## 0. Demo trên video CCTV thật (Phố Huế – Trần Khát Chân) ✅ HOÀN THÀNH (2026-06-22)

**Chi tiết đầy đủ**: `docs/demo.md` mục 8.

Đã dựng map thật (`Map/pho_hue_tkc.*`) + calibration camera best-effort + script demo mới
(`custom_tracking_system/scripts/demo_real_video_scenario.py`) chạy detect+track+Goal
Classifier+Route Predictor THẬT trên 1 video CCTV thật, không qua CARLA. Đã verify
end-to-end (track world-position hợp lý, đánh dấu tay tai nạn qua API, route 4-hop ra
đúng trên map thật).

**Còn lại trước khi demo thật trước hội đồng:**
- [ ] Rehearsal trực tiếp 2-3 lần (chưa làm) — theo checklist Phase 4 đã thống nhất
- [ ] (Tuỳ chọn) Tăng độ chính xác calibration bằng đo đạc thực địa hoặc click điểm
      chuẩn qua Street View, thay cho cách khớp bằng mắt hiện tại
- [ ] (Tuỳ chọn) Thêm nút "Mark accident" trên `map_view.html` gọi thẳng API, thay vì
      phải dùng curl/Postman thủ công
- [ ] (Tuỳ chọn) Test thêm 1-2 video khác cùng giao lộ Phố Huế để xác nhận calibration
      ổn định khi đổi video

---

## 0b. Route Predictor: confidence thật cho hop 2-4 + fix bug "lãng phí" hướng rẽ ✅ HOÀN THÀNH (2026-06-22)

**Chi tiết đầy đủ**: `docs/demo.md` mục 9.

Phát hiện qua phân tích `data/trajectories_hanoi_v2/episode_*_goals.csv` (36,997 goal events,
2,263 chuỗi xe đi qua ≥2 junction): giả định cũ "luôn đi thẳng" cho hop≥1 chỉ đúng **37.04%**
trên dữ liệu thật — không có cơ sở để báo confidence=100% như trước. Đã thay bằng prior thực
nghiệm (`custom_tracking_system/data/route_predictor_priors.json`, script tạo:
`scripts/eval/eval_route_predictor_priors.py`). Đồng thời fix 1 bug: hướng rẽ thật từ Goal
Classifier áp dụng theo **chỉ số loop** (hop==0) thay vì "junction thật đầu tiên gặp" — với
map nhiều connector ngắn (như pho_hue_tkc), hướng rẽ thật bị gán nhầm vào 1 đoạn không phải
junction, coi như bị bỏ qua. Đã sửa dùng cờ `direction_applied` thay cho check `hop==0`.

**Còn lại / hướng mở rộng (tuỳ chọn):**
- [ ] Markov bậc 1 (dựa hướng hop trước) chỉ nhích từ 37.0% → 39.0% — không đáng để dùng
      thay prior marginal đơn giản, nhưng nếu có thời gian có thể thử thêm road-type
      (đường lớn/nhỏ) làm điều kiện, may ra tốt hơn Markov theo hướng rẽ thuần.
- [ ] Áp dụng tương tự cho dữ liệu CCTV thật nếu sau này có chuỗi multi-junction thật
      (hiện dataset real-world chỉ có snapshot 1 junction/clip, không đủ để tính).

---

## 0c. Goal Classifier real-world: domain gap CARLA vs real ✅ ĐÃ XỬ LÝ PHẦN ACTIONABLE (2026-06-22)

**Phát hiện**: feature space CARLA (world-space, m/s) và real (pixel-space, px/s + bbox)
**khác hoàn toàn nhau** (đọc trực tiếp `train_goal_classifier.py` vs `train_goal_classifier_real.py`)
→ không thể "fine-tune" 1 model sang model khác như từng giả định. Domain gap 70.3% (CARLA)
vs 59.7% (real) phản ánh đúng độ khó khác nhau của 2 bài toán, không phải thiếu sót có thể
fix bằng transfer learning đơn giản.

**Đã thử (so sánh công bằng trên cùng data, cùng train/test split)**:

| Model | Accuracy | Balanced acc | F1 macro | ECE | u_turn recall |
|---|---|---|---|---|---|
| GB, horizon=1.0s (cũ) | 59.73% | 51.96% | 0.519 | 0.116 | 34.5% |
| RandomForest, horizon=1.0s | 55.72% | 51.65% | 0.488 | 0.021 | 44.8% |
| MLP, horizon=1.0s | 54.92% | 44.24% | 0.432 | 0.047 | 13.8% |
| **GB, horizon=1.5s (MỚI — đã deploy production)** | **60.12%** | **54.18%** | **0.523** | 0.101 | 36.1% |

**Phát hiện quan trọng + đã fix**: `goal_classifier.py:47` (inference) đã hardcode
`_HORIZON = 1.5` với comment "best per-horizon result from training" từ trước — nhưng
weights production thực tế lại được train với `--horizon 1.0` (default của script train).
Đây là **mismatch thật giữa train-time và inference-time** đã tồn tại sẵn (không phải tôi
gây ra). Đã train lại đúng `--horizon 1.5`, deploy vào `weights/goal_classifier_real.pkl`
(bản cũ backup tại `weights/_backup_horizon1.0/`), verify lại load + chạy demo OK.

**Chưa làm / còn mở** (không bắt buộc trước demo):
- [ ] RandomForest có calibration tốt hơn nhiều (ECE 0.021) và u_turn recall cao nhất
      (44.8%) — nếu ưu tiên "confidence đáng tin" hơn accuracy thô, có thể đổi sang RF.
      Hiện vẫn giữ GB vì accuracy/f1 tổng thể cao hơn.
- [ ] LSTM trên raw sequence (chưa thử — cần viết pipeline mới, không chỉ đổi `--model`).
- [ ] Tạo data tổng hợp pixel-space từ CARLA qua camera ảo (ý tưởng lớn hơn, chưa làm).

---

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
