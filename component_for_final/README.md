# component_for_final

Tập hợp **hình vẽ thiết kế (UML)** và **hình ảnh sản phẩm/kết quả** để chèn vào báo
cáo đồ án tốt nghiệp. Tổ chức theo 3 phần:

```
component_for_final/
├── uml/              # 13 file PlantUML (.puml) — sơ đồ thiết kế
├── figures_results/  # 13 ảnh kết quả/sản phẩm có sẵn (model output, bản đồ, footage)
└── screenshots/      # script + hướng dẫn tự chụp ảnh giao diện đang chạy (live UI)
```

---

## 1. UML — `uml/*.puml`

Chưa render sẵn ra PNG vì máy **không có Java** (PlantUML jar cần JRE). Cách render:

- **VS Code**: cài extension *PlantUML* (jebbs) → mở file `.puml` → `Alt+D` để xem,
  chuột phải → *Export Current Diagram* → PNG/SVG. (Extension này vẫn cần Java; hoặc
  đặt `plantuml.render = PlantUMLServer` để render qua server không cần Java.)
- **Online**: dán nội dung vào https://www.plantuml.com/plantuml (lưu ý: gửi nội
  dung lên dịch vụ ngoài).
- **CLI** (nếu cài Java): `java -jar plantuml.jar uml/*.puml`

| File | Loại sơ đồ | Hình trong báo cáo (đề xuất) |
|---|---|---|
| `01_usecase_tong_quat.puml` | Use case tổng quát | Hình 2.1 |
| `02_usecase_uc1_phat_hien_theo_doi.puml` | Use case phân rã UC1 | Hình 2.2 |
| `03_usecase_uc2_xuyen_camera.puml` | Use case phân rã UC2 | Hình 2.3 |
| `04_usecase_uc3_su_co.puml` | Use case phân rã UC3 | Hình 2.4 |
| `05_usecase_uc4_hanh_trinh.puml` | Use case phân rã UC4 | Hình 2.5 |
| `06_activity_quy_trinh_truy_vet.puml` | Activity - quy trình nghiệp vụ | Hình 2.6 |
| `07_activity_pipeline_1frame.puml` | Activity - xử lý 1 khung hình | Ch3/Ch4 |
| `08_sequence_tracking_xuyen_camera.puml` | Sequence - tracking xuyên camera | Ch4 |
| `09_sequence_danh_dau_route_map.puml` | Sequence - đánh dấu → route → map | Ch4 |
| `10_class_diagram.puml` | Class diagram | Ch4 - thiết kế lớp |
| `11_package_diagram.puml` | Package diagram (phân tầng) | Ch4 - thiết kế tổng quan |
| `12_component_kien_truc_tong_the.puml` | Component - kiến trúc tổng thể | Hình 1.1 / Ch4 |
| `13_deployment.puml` | Deployment diagram | Ch4 - triển khai |

> Các sơ đồ use case/activity được dựng **khớp đúng** mô tả trong
> `docs/chuong2_khao_sat_phan_tich_yeu_cau.md`; class/package/component/deployment
> dựng theo cấu trúc code thật (`custom_tracking_system/`, `server/`, `frontend/`).

---

## 2. Hình kết quả / sản phẩm — `figures_results/*`

Ảnh **có sẵn, là output thật** của hệ thống (không phải UI cần chạy live).

| File | Nội dung | Hình trong báo cáo (đề xuất) |
|---|---|---|
| `01_ban_do_hanoi_district3_topdown.png` | Bản đồ Hà Nội (Mỗ Lao) dựng trong CARLA | Ch4 - pipeline bản đồ |
| `02_cctv_pho_hue_sample_frame.jpg` | Khung hình CCTV thật (Phố Huế – Trần Khát Chân) | Ch2/Ch4 - dữ liệu thật |
| `03_cctv_pho_hue_calibration_overlay.jpg` | Hiệu chuẩn pixel↔world chồng lên footage thật | Ch4/Ch5 |
| `04_pho_hue_so_do_giao_lo.png` | Sơ đồ giao lộ Phố Huế (junction graph) | Ch4 |
| `05_detection_yolo11s_vn_predictions.jpg` | Kết quả detect YOLO11s (val batch) | Ch4 - kết quả detection |
| `06_detection_yolo11s_vn_confusion.png` | Confusion matrix detection | Ch5 - đánh giá detection |
| `07_detection_yolo11s_vn_PR_curve.png` | Precision-Recall curve | Ch5 |
| `08_reid_osnet_veri776_cmc_curve.png` | CMC curve ReID (VeRi-776) | Ch5 - đánh giá ReID |
| `09_reid_osnet_veri776_results.png` | Bảng/biểu kết quả ReID | Ch5 |
| `10_tracking_mota_idf1.png` | MOTA/IDF1 theo camera | Ch5 - đánh giá tracking |
| `11_goal_classifier_confusion.png` | Confusion matrix dự đoán hướng rẽ | Ch5 |
| `12_goal_classifier_horizon_accuracy.png` | Accuracy theo horizon dự đoán | Ch5 |
| `13_goal_classifier_class_accuracy.png` | Accuracy theo lớp | Ch5 |

Nguồn gốc (để truy vết): các ảnh lấy từ `Map/`, `docs/assets/yolo11s_vn/`,
`docs/assets/osnet_veri776_v2/`, `docs/assets/goal_classifier_real/`,
`custom_tracking_system/docs/assets/mot_eval/`.

---

## 3. Ảnh giao diện sản phẩm (live UI) — `screenshots/`

**Chưa tạo sẵn** — cần chạy CARLA + server + frontend rồi tự chụp. Xem
`screenshots/README.md` (có sẵn script Playwright `capture_ui.mjs`). Cần bổ sung:

- `14_ui_dashboard_camera_grid.png` — Dashboard giám sát đa camera
- `15_ui_mapview_route_prediction.png` — Bản đồ + hành trình dự đoán
- `16_ui_alert_management.png` — Quản lý cảnh báo/sự cố

---

## Ghi chú quan trọng trước khi đưa vào báo cáo

1. **Render UML ra PNG/SVG** rồi mới chèn (file `.puml` là mã nguồn, không phải ảnh).
2. **Chụp 3 ảnh UI** còn thiếu ở mục 3.
3. **Thống nhất số liệu** giữa các bản nháp trước khi viết phần kết quả (YOLO11s vs
   11m; mAP trên tập nào; OSNet đã có benchmark v2 — bỏ ghi chú "mất dataset" cũ).
