# Ảnh chụp giao diện sản phẩm (live UI)

Thư mục này chứa script tự động chụp ảnh giao diện đang chạy. **Các ảnh UI trực tiếp
(Dashboard, MapView, Alert Management) chưa được tạo sẵn** vì cần chạy đồng thời
CARLA + server + frontend (có GPU + màn hình) để giao diện có dữ liệu thật — không
tái lập được trong môi trường tự động. Hãy tự chụp theo hướng dẫn dưới.

## Cách tạo ảnh UI

1. Khởi động toàn bộ hệ thống (để có dữ liệu thật trên giao diện):
   ```bat
   cd e:\finalproject\AI_custom
   start.bat
   ```
   (CARLA → server `--with-ai` → frontend dev :3000). Chờ tới khi mở được
   `http://localhost:3000`. Để demo bản đồ: vào tab **Bản đồ**, click một xe và
   chọn **"Đánh dấu theo dõi"** để route hiện lên trước khi chụp.

2. Chụp tự động bằng Playwright:
   ```bat
   cd e:\finalproject\AI_custom\component_for_final\screenshots
   npm init -y
   npm i -D playwright
   npx playwright install chromium
   node capture_ui.mjs
   ```

Ảnh xuất ra trong chính thư mục này:

| File | Trang | Dùng cho |
|---|---|---|
| `14_ui_dashboard_camera_grid.png` | `/` | Ch4 - minh hoạ chức năng giám sát đa camera |
| `15_ui_mapview_route_prediction.png` | `/map` | Ch4/Ch5 - vẽ hành trình dự đoán lên bản đồ |
| `16_ui_alert_management.png` | `/alerts` | Ch4 - quản lý cảnh báo/sự cố |

## Chụp thủ công (nếu không dùng Playwright)

Mở từng trang trên trình duyệt và dùng công cụ chụp màn hình của Windows
(`Win + Shift + S`), lưu đúng tên file ở bảng trên vào thư mục này.

> Mẹo cho báo cáo: nên chụp thêm 1 ảnh lúc có **sự cố CRITICAL** (panel cảnh báo
> nhấp nháy) và 1 ảnh **camera feed có bbox + Global ID + mũi tên hướng rẽ** để
> minh hoạ rõ các đóng góp ở Chương 5.
