# Bằng chứng hand-verify Goal Classifier (accuracy thật)

*Sinh tự động 2026-07-12 20:12 bởi `verify_goal_labels.py`.*

## Phương pháp
- Tái tạo ĐÚNG test set held-out (1.334 track, seed=42) — tách hoàn toàn khỏi train.
- Lấy mẫu phân tầng ngẫu nhiên, **200 track** được người kiểm xác nhận hướng đi thật
  bằng cách xem quỹ đạo vẽ đè lên khung hình CCTV thật.
- So dự đoán model với nhãn NGƯỜI (không phải nhãn auto-label nhiễu).

## Kết quả

| Chỉ số | Giá trị |
|---|---|
| **Accuracy thật (model vs người)** | **0.650**  (95% CI 0.582–0.713) |
| Nhãn auto-label đúng (auto vs người) | 0.755 (độ nhiễu 24.5%) |
| Model vs auto (sanity) | 0.705 |

### Recall theo lớp (vs nhãn người)

| Hướng | n | Recall | 95% CI |
|---|---|---|---|
| straight | 153 | 0.627 | 0.549–0.700 |
| left | 32 | 0.719 | 0.546–0.844 |
| right | 14 | 0.714 | 0.454–0.883 |
| u_turn | 1 | 1.000 | 0.207–1.000 |

## Tệp kèm theo (bằng chứng)
- `metrics.json` — toàn bộ số liệu máy đọc được.
- `confusion_matrix_human.png` — ma trận nhầm lẫn model vs nhãn người.
- `montage_verified_tracks.png` — ảnh mẫu các track đã kiểm (OK=đúng, MISS=sai).
- `overlays/` — ảnh từng track đã kiểm (dấu vết kiểm chứng đầy đủ).
- `../../custom_tracking_system/data/goal_labels_verified.csv` — nhãn người từng track (tái lập được).