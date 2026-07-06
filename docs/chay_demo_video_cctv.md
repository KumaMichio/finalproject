# Chạy demo trên video CCTV (prototype máy đơn)

Tài liệu này mô tả cách chạy **toàn bộ hệ thống trên một máy đơn**, dùng **video
CCTV ghi sẵn** làm nguồn (không cần CARLA, không cần camera live). Đây là phạm vi
demo/prototype của đồ án. Mục tiêu: một người khác (hoặc hội đồng) tái lập được
kết quả từ đầu.

> Nguồn video hỗ trợ qua lớp `VideoSource`: **file** (đã kiểm chứng), webcam,
> RTSP. Prototype này chỉ dùng và kiểm chứng nguồn **file**.

---

## 1. Yêu cầu

| Thành phần | Giá trị |
|---|---|
| OS | Windows 10/11 64-bit |
| GPU | NVIDIA (khuyến nghị) — CPU vẫn chạy nhưng chậm |
| Python (AI + backend) | `custom_tracking_system/venv_tracking` (Python 3.10, đã cài sẵn torch/ultralytics/boxmot/fastapi) |
| Node.js | 18+ (frontend) |
| Model weights | `custom_tracking_system/weights/yolo11s_vn.pt`, `goal_classifier_real.pkl` (+ scaler/encoder/feature json) — phải có sẵn |

Kiểm tra nhanh weights:

```bash
ls custom_tracking_system/weights/yolo11s_vn.pt custom_tracking_system/weights/goal_classifier_real.pkl
```

Ký hiệu interpreter dùng suốt tài liệu:

```
PY = custom_tracking_system\venv_tracking\Scripts\python.exe
```

---

## 2. Chạy (2 cửa sổ)

### Cửa sổ 1 — Backend + AI pipeline trên video CCTV

```cmd
cd e:\School_project\finalproject\server

..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py ^
  --with-ai ^
  --source file ^
  --video-path "..\custom_tracking_system\data\datasets\traffic_intersection\02. Dữ liệu camera giao thông\Phố Huế Trần Khát Chân\12h.10.9.22.mp4" ^
  --camera-ids CAM_PHO_HUE_REAL ^
  --config ..\custom_tracking_system\config\camera_config_pho_hue.yaml ^
  --loop-video
```

- `--video-path` lặp lại được để chạy nhiều luồng (mỗi `--video-path` một
  camera); `--camera-ids` liệt kê ID theo đúng thứ tự, phân tách bằng dấu phẩy.
- Config `camera_config_pho_hue.yaml` bật `route_prediction` (bản đồ Phố Huế –
  Trần Khát Chân) và khai báo camera `CAM_PHO_HUE_REAL`, vùng làn, tọa độ đèn.
- Server sẵn sàng ở `http://localhost:8000` (Swagger: `/docs`).

### Cửa sổ 2 — Frontend

```cmd
cd e:\School_project\finalproject\frontend
npm install   # chỉ lần đầu
npm run dev
```

Mở **http://localhost:3000**.

---

## 3. Kết quả mong đợi

- **Dashboard** (`/`): luồng camera với bounding box + track ID ổn định; vi phạm
  "đi sai làn" được khoanh đỏ và đẩy vào Alert Feed bên phải.
- **Bản đồ** (`/map`): chọn một phương tiện → hiện vị trí trên bản đồ Phố Huế và
  vector hướng đi dự đoán kèm độ tin cậy.
- **Cấu hình ROI** (`/config`): xem mục 4.

---

## 4. Chỉnh vùng quan tâm (ROI) trên UI — UC4.2

Tính năng cho phép người vận hành **vẽ/sửa/xóa vùng ngay trên giao diện**, áp
dụng vào pipeline **tức thì, không cần restart**.

1. Trên Dashboard bấm **"Cấu hình ROI"** (hoặc mở `/config`).
2. Chọn camera. Ảnh nền là một khung hình hiện tại của camera đó
   (`GET /stream/{cam}/snapshot`).
3. Chọn loại vùng:
   - **Làn (sai làn)** — chọn lớp phương tiện được phép + hướng hợp lệ. Dùng cho
     phát hiện `LANE_VIOLATION`.
   - **Vùng cảnh báo** — chọn loại (entry/exit/loiter/speed).
4. **Click lên ảnh** để thêm các đỉnh polygon (≥ 3 điểm), đặt tên, bấm **Lưu vùng**.
5. Vùng mới có hiệu lực ngay: quay lại Dashboard, phương tiện đi sai vùng làn vừa
   vẽ sẽ bị bắt lỗi mà **không cần khởi động lại server**.

Cơ chế: `POST/PUT/DELETE /api/rois` ghi vào SQLite rồi gọi
`AIProcessor.reload_rois_from_db()` → `apply_rois()` đẩy nóng vào
`alert_system` (vùng cảnh báo) và `incident_detector` (vùng làn).

---

## 5. Ghi chú về cấu hình ROI (nguồn sự thật)

- **DB là nguồn sự thật** cho ROI. Lần đầu chạy, hệ thống **seed** ROI từ file
  YAML (`camera_config_*.yaml`) vào bảng `rois` (SQLite). Sau đó mọi chỉnh sửa
  đều qua UI/DB; chỉnh YAML về sau sẽ **không** ghi đè DB nữa.
- Toạ độ polygon lưu theo **độ phân giải gốc** của camera (khai báo trong config).
- **Reset ROI về YAML**: xóa các dòng trong bảng `rois` (hoặc xóa
  `tracking_system.db`) rồi khởi động lại — hệ thống seed lại từ YAML.

---

## 6. Sự cố thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `/config` báo "Chưa có khung hình" khi lấy snapshot | Đợi backend `--with-ai` nạp xong pipeline và có frame đầu tiên; bấm "⟳ Làm mới ảnh". |
| Không thấy bản đồ | Config phải có section `route_prediction` (dùng `camera_config_pho_hue.yaml`). |
| Vi phạm đèn đỏ / vượt vạch dừng không kích hoạt trên video Phố Huế | Trên camera này chỉ bật "đi sai làn"; hai vi phạm còn lại minh họa trên video kịch bản có cấu hình đèn/vạch tương ứng. |
| FPS thấp | Giảm `resolution` trong config, hoặc chạy ít camera hơn. |
