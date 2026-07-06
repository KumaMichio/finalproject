# Checklist diễn tập demo trước hội đồng

Mục tiêu: chạy được **toàn bộ sản phẩm end-to-end** trên máy GPU demo (NVIDIA
RTX 4060/3060) và trình diễn đủ **4 use case** khớp báo cáo, kèm các *caveat
trung thực* đã ghi trong Chương 5–6. Đây là bản diễn tập cho mục 6.1 ("demo chưa
diễn tập trực tiếp trước hội đồng").

> Máy dev hiện tại (AMD RX6700XT, torch CPU-only) **không** phải máy demo — chạy
> được nhưng rất chậm. Mọi số FPS/độ mượt phải đo trên máy GPU.

---

## 0. Tiền kiểm (làm trước ngày demo)

- [ ] `custom_tracking_system/venv_tracking` đầy đủ (torch CUDA, ultralytics, boxmot, fastapi).
- [ ] `nvidia-smi` thấy GPU; torch nhận CUDA:
      `venv_tracking\Scripts\python.exe -c "import torch;print(torch.cuda.is_available())"` → `True`.
- [ ] Weights có mặt: `yolo11s_vn.pt`, `goal_classifier_real.pkl` (+ `goal_scaler_real.pkl`,
      `goal_label_encoder_real.pkl`, `goal_feature_names_real.json`).
- [ ] Video demo có mặt: clip Phố Huế – Trần Khát Chân (vd `12h.10.9.22.mp4`).
- [ ] Map: `Map/pho_hue_tkc_junction_graph.json` + `.xodr` có mặt.
- [ ] Frontend: `cd frontend && npm install` (một lần), `npm run build` không lỗi.
- [ ] **Reset trạng thái sạch** (tránh ROI/alert cũ gây nhiễu): xoá `tracking_system.db`
      để hệ thống seed lại ROI từ YAML, và dọn `evidence/` nếu cần.

---

## 1. Khởi động (2 cửa sổ)

**Cửa sổ 1 — Backend + AI trên video Phố Huế:**

```cmd
cd e:\School_project\finalproject\server
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py --with-ai ^
  --source file ^
  --video-path "..\custom_tracking_system\data\datasets\traffic_intersection\02. Dữ liệu camera giao thông\Phố Huế Trần Khát Chân\12h.10.9.22.mp4" ^
  --camera-ids CAM_PHO_HUE_REAL ^
  --config ..\custom_tracking_system\config\camera_config_pho_hue.yaml ^
  --loop-video
```

**Cửa sổ 2 — Frontend:** `cd frontend && npm run dev` → mở `http://localhost:3000`.

- [ ] Backend log: nạp YOLO + Goal Classifier + Route Predictor, không traceback.
- [ ] `http://localhost:8000/docs` mở được (Swagger).
- [ ] Dashboard hiện luồng camera có bounding box + track ID.

---

## 2. Kịch bản trình diễn theo use case

### UC1 — Phát hiện & theo dõi (YOLO11s_vn + ByteTrack)
- [ ] Dashboard: box bám phương tiện, **track ID ổn định** khi xe di chuyển.
- [ ] Chỉ cho hội đồng thấy ID **không nhảy** khi xe bị che một phần rồi hiện lại.
- Số liệu nói kèm: mAP50 = 89,5% (caveat: đo lại trên pool huấn luyện — Ch5.1/6.1);
  IDF1 ≈ 100% trên Phố Huế (caveat cấu trúc GT bán tự động → MOTA bị thổi phồng, Ch5.4).

### UC2 — Phát hiện vi phạm (hệ luật minh bạch)
- [ ] **Đi sai làn (`LANE_VIOLATION`)** — headline trên video thật: xe đi vào sai
      vùng làn (ô tô lấn làn xe máy / ngược lại) bị **khoanh đỏ** + đẩy vào Alert Feed.
- Caveat trung thực (khớp Ch6.1): **vượt đèn đỏ/vượt vạch dừng tạm TẮT** trên clip
  Phố Huế (đèn chỉ Red khung ~84–95 rồi tắt hẳn; xe dừng đúng đèn nên không có
  positive thật để minh hoạ). Logic đã có sẵn, đúng; re-enable là future work (mục 4).

### UC3 — Dự đoán hướng đi dài hạn trên bản đồ (đóng góp chính)
- [ ] `/map`: chọn 1 phương tiện → hiện **vị trí trên bản đồ Phố Huế thực** +
      **vector hướng đi dự đoán** kèm **độ tin cậy**.
- [ ] Nhấn mạnh **tính minh bạch**: ngã tư xa KHÔNG hiển thị "100%" mà theo prior
      (Ch5.3) — thà thừa nhận bất định còn hơn phóng đại.
- Số liệu: accuracy 60,12%, balanced acc 54,18% trên 1.334 track kiểm thử tách
  riêng (Ch5.2). Caveat: recall "quay đầu" 36% / chỉ 36 mẫu (Ch6.1).

### UC4 — Cấu hình: vẽ/sửa ROI trên UI (nóng, không restart)
- [ ] Dashboard → **"Cấu hình ROI"** (`/config`) → chọn camera → ảnh nền là snapshot hiện tại.
- [ ] Vẽ 1 vùng **Làn** mới (≥3 đỉnh) + chọn lớp phương tiện hợp lệ → **Lưu**.
- [ ] Quay lại Dashboard: phương tiện đi sai vùng làn vừa vẽ **bị bắt lỗi ngay**,
      **không restart server** (chứng minh hot-reload: `POST /api/rois` → `reload_rois_from_db`).

### CN#2 — Đánh dấu đối tượng nghi vấn
- [ ] `/map`: **đánh dấu** (mark) một phương tiện → hệ thống lưu **snapshot** đối tượng
      (`marked_snapshot_path`) và tô đậm hành trình dự đoán của đúng đối tượng đó.
- Caveat: **lưu biển số** là future work (Ch6.2.3) — số liệu OCR còn thấp
  (xe máy 14–33px, dưới sàn cảm biến), *cố ý chưa tích hợp*.

---

## 3. Đo FPS 6 camera (mục tiêu ~15–20 FPS, Ch6.2.2)

Khởi động backend với **6 luồng** (lặp video Phố Huế ×6, đúng độ phân giải thật):

```cmd
cd server
..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py --with-ai --source file ^
  --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" ^
  --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" ^
  --camera-ids CAM1,CAM2,CAM3,CAM4,CAM5,CAM6 ^
  --config ..\custom_tracking_system\config\camera_config_pho_hue.yaml --loop-video
```

Đợi ~15s rồi đo:

```cmd
cd custom_tracking_system\scripts\eval
..\..\venv_tracking\Scripts\python.exe measure_6cam_fps.py --duration 60 --warmup 15 --out fps_6cam.json
```

- [ ] Ghi lại `fps_median` + `active_cameras=6` vào báo cáo/slide (thay cho câu
      "chưa đo" ở Ch6.1). Nếu chưa đạt 15 FPS: giảm `resolution` trong config hoặc
      bật batch/điều tiết tần suất (đã có — Ch4.5).

---

## 4. (Tuỳ chọn) Bật lại red-light để đi xa hơn báo cáo

Chỉ làm sau khi **verify sạch** trên máy GPU:
1. Trong `camera_config_pho_hue.yaml`, xoá dòng `- RED_LIGHT_VIOLATION` khỏi `disabled_types`.
2. Chạy 1 phút clip đường thoáng, **đối chiếu từng cảnh báo red-light với mắt**:
   không có false-positive (bộ phân loại đọc 1 điểm pixel — rủi ro misread).
3. Nếu sạch → có thể trình diễn như "vượt kỳ vọng báo cáo". Nếu bẩn → giữ TẮT.

---

## 5. Sự cố thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `/config` báo "Chưa có khung hình" | Đợi pipeline nạp xong frame đầu; bấm "⟳ Làm mới ảnh". |
| Không thấy bản đồ ở `/map` | Config phải có `route_prediction` (dùng `camera_config_pho_hue.yaml`). |
| Red-light/vạch dừng không kích hoạt | Đúng như thiết kế — đang TẮT trên Phố Huế (xem UC2 caveat). |
| FPS thấp | Giảm `resolution`, chạy ít camera hơn, kiểm tra torch thấy CUDA. |
| Alert cũ gây nhiễu | Xoá `tracking_system.db` + `evidence/` trước khi demo. |
