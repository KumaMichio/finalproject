# Mô tả Workspace CARLA 0.9.9.4

## Tổng quan
Workspace này chứa phiên bản CARLA 0.9.9.4, một simulator mã nguồn mở cho việc phát triển và thử nghiệm xe tự lái (autonomous driving). CARLA cung cấp môi trường mô phỏng thực tế với các thành phố ảo, phương tiện, và cảm biến để hỗ trợ nghiên cứu AI và robot.

## Cấu trúc Folder

### Root Files
- **instruction.txt**: Tài liệu hướng dẫn sử dụng workspace này.
- **plan.md**: Kế hoạch phát triển hoặc ghi chú về dự án.
- **README.md**: Tài liệu giới thiệu tổng quan về CARLA và cách cài đặt/sử dụng.

### PythonAPI/
Thư mục chứa API Python để tương tác với CARLA simulator.
- **custom_tracking_system/**: Hệ thống theo dõi tùy chỉnh.
  - **__init__.py**: File khởi tạo module Python.
  - **main.py**: File chính để chạy hệ thống theo dõi.
  - **README.md**: Tài liệu hướng dẫn cho hệ thống theo dõi.
  - **requirements.txt**: Danh sách các thư viện Python cần thiết.
  - **run.bat**: Script batch để chạy trên Windows.
  - **config/**: Thư mục cấu hình.
    - **camera_config.yaml**: Cấu hình camera.
  - **datasets/**: Dữ liệu cho hệ thống.
    - **ground_truth/**: Dữ liệu ground truth.
    - **synthetic_data/**: Dữ liệu tổng hợp.
  - **models/**: Mô hình AI/ML.
    - **detection/**: Mô hình phát hiện đối tượng.
    - **reid/**: Mô hình nhận dạng lại (Re-identification).
    - **tracking/**: Mô hình theo dõi.
  - **modules/**: Các module chức năng.
    - **alert_system.py**: Hệ thống cảnh báo.
    - **camera_controller.py**: Điều khiển camera.
    - **detector.py**: Module phát hiện.
    - **global_tracking.py**: Theo dõi toàn cục.
    - **reid.py**: Module Re-ID.
    - **tracker.py**: Module theo dõi.
    - **traffic_generator.py**: Tạo lưu lượng giao thông.
    - **trajectory_predictor.py**: Dự đoán quỹ đạo.
  - **utils/**: Các tiện ích.
    - **data_writer.py**: Ghi dữ liệu.
    - **metrics.py**: Tính toán metrics.
    - **visualization.py**: Trực quan hóa dữ liệu.
- **carla/**: Thư mục chứa thư viện CARLA Python API.
- **examples/**: Các ví dụ sử dụng CARLA API.
- **util/**: Các tiện ích bổ sung.

### WindowsNoEditor/
Thư mục chứa các file thực thi và cấu hình cho CARLA trên Windows (không có editor Unreal Engine).
- **CHANGELOG**: Nhật ký thay đổi phiên bản.
- **Dockerfile**: File để build Docker image.
- **LICENSE**: Giấy phép sử dụng.
- **README**: Tài liệu README cho phiên bản Windows.
- **CarlaUE4/**: Dự án Unreal Engine cho CARLA.
  - **CarlaUE4.uproject**: File dự án Unreal Engine.
  - **Binaries/Win64/**: File thực thi nhị phân cho Windows 64-bit.
  - **Config/**: Cấu hình game/engine.
    - **DefaultEngine.ini**: Cấu hình engine mặc định.
    - **DefaultGame.ini**: Cấu hình game mặc định.
    - **DefaultGameUserSettings.ini**: Cài đặt người dùng game.
    - **DefaultInput.ini**: Cấu hình input.
  - **Content/Carla/**: Nội dung game CARLA.
  - **Plugins/Carla/**: Plugin CARLA cho Unreal Engine.
- **Co-Simulation/**: Tích hợp với các simulator khác.
  - **PTV-Vissim/**: Tích hợp với PTV Vissim.
    - **run_synchronization.py**: Script đồng bộ hóa.
    - **data/**: Dữ liệu cho tích hợp.
    - **examples/**: Ví dụ sử dụng.
    - **vissim_integration/**: Module tích hợp Vissim.
  - **Sumo/**: Tích hợp với SUMO (Simulation of Urban MObility).
    - **requirements.txt**: Thư viện cần thiết.
    - **run_synchronization.py**: Script đồng bộ hóa.
    - **spawn_npc_sumo.py**: Script tạo NPC từ SUMO.
    - **data/**: Dữ liệu.
    - **examples/**: Ví dụ.
    - **sumo_integration/**: Module tích hợp SUMO.
    - **util/**: Tiện ích.
- **Engine/**: Engine Unreal Engine.
  - **Binaries/ThirdParty/**: Thư viện bên thứ ba.
  - **Config/**: Cấu hình engine.
    - **Base*.ini**: Các file cấu hình cơ bản.
    - **Layouts/**: Bố cục.
    - **Windows/**: Cấu hình Windows-specific.
  - **Content/**: Nội dung engine (materials, meshes, sounds, etc.).
    - **Animation/**: Animation assets.
    - **ArtTools/**: Công cụ nghệ thuật.
    - **BasicShapes/**: Hình dạng cơ bản.
    - **Editor*/**: Tài nguyên editor.
    - **Engine*/**: Tài nguyên engine.
    - **Functions/**: Hàm.
    - **Internationalization/**: Quốc tế hóa.
    - **Localization/**: Địa phương hóa.
    - **Maps/**: Bản đồ.
    - **MapTemplates/**: Mẫu bản đồ.
    - **MobileResources/**: Tài nguyên mobile.
    - **Slate/**: UI framework.
    - **Tutorial/**: Hướng dẫn.
    - **VREditor/**: VR Editor.
  - **Plugins/**: Các plugin engine (2D, AI, Blendables, Developer, Editor, Enterprise, Experimental, MagicLeap, Media, etc.).

### HDMaps/
- **README**: Tài liệu về High Definition Maps.

### Co-Simulation/
(Tương tự như trong WindowsNoEditor, có thể là duplicate hoặc symlink).

### Engine/
(Tương tự như trong WindowsNoEditor).