# Build CARLA từ UE4 Source trên Windows

> **Mục đích**: Hướng dẫn build CARLA từ source để có UE4 Editor — từ đó import
> custom 3D asset (tòa nhà Mỗ Lao, xe máy VN, v.v.) và đóng gói thành build
> packaged riêng thay thế `WindowsNoEditor` hiện tại.
>
> **Lý do cần**: Build packaged `WindowsNoEditor` hiện tại (CARLA 0.9.14) bị giới hạn
> `-quality-level=Low` → mesh xe + prop không render ở camera gần (~30-40m).
> Không thể import asset mới vào build packaged — phải có UE4 Editor.
>
> **Cảnh báo phần cứng**: Máy hiện tại có 16GB RAM — thấp hơn yêu cầu 32GB+.
> Xem §7 để biết cách workaround và rủi ro.

---

## Mục lục

1. [Yêu cầu](#1-yêu-cầu)
2. [Cấp quyền truy cập UE4 source](#2-cấp-quyền-truy-cập-ue4-source)
3. [Clone và build Unreal Engine 4.26](#3-clone-và-build-unreal-engine-426)
4. [Clone và setup CARLA](#4-clone-và-setup-carla)
5. [Build CARLA lần đầu](#5-build-carla-lần-đầu)
6. [Import custom 3D asset vào CARLA Editor](#6-import-custom-3d-asset-vào-carla-editor)
7. [Workaround 16GB RAM](#7-workaround-16gb-ram)
8. [Đóng gói thành packaged build](#8-đóng-gói-thành-packaged-build)
9. [Tích hợp với pipeline hiện tại](#9-tích-hợp-với-pipeline-hiện-tại)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Yêu cầu

### 1.1 Phần cứng

| Thành phần | Tối thiểu | Khuyến nghị | Máy hiện tại |
|---|---|---|---|
| RAM | **32 GB** | 64 GB | 16 GB ⚠️ (xem §7) |
| Disk (build UE4 + CARLA) | **100 GB** | 150 GB | Cần kiểm tra |
| GPU | DX11+ (bất kỳ) | RTX 4060+ | RTX 4060 ✅ |
| CPU | 8 thread+ | 16 thread+ | i5-9300H 8T ✅ (chậm) |

> **Disk breakdown**: UE4 source ~15GB + build output ~50GB + CARLA source ~5GB +
> CARLA assets (tải về) ~30GB + UE4 Editor temp ~20GB = ~120GB tổng.

### 1.2 Phần mềm (Windows 10/11)

| Phần mềm | Phiên bản bắt buộc | Link |
|---|---|---|
| **Visual Studio 2019** | Community hoặc Professional | visualstudio.microsoft.com |
| **Windows 10 SDK** | 10.0.18362 trở lên | Cài trong VS2019 Installer |
| **CMake** | >= 3.15 | cmake.org |
| **Git** | bất kỳ | git-scm.com |
| **Python** | **3.8** (không phải 3.10) | python.org — cần riêng cho build scripts |
| **7-Zip** | bất kỳ | 7-zip.org |
| **aria2** | bất kỳ | aria2.github.io (tăng tốc download assets) |
| **Make** | via Chocolatey | `choco install make` |

**Cài Visual Studio 2019 Workloads cần thiết:**
- ✅ Desktop development with C++
- ✅ Windows 10 SDK (10.0.18362.0 hoặc mới hơn)
- ✅ MSVC v142 - VS 2019 C++ x64/x86 build tools
- ✅ C++ CMake tools for Windows (optional nhưng tiện)

---

## 2. Cấp quyền truy cập UE4 source

UE4 source code yêu cầu tài khoản Epic Games và link với GitHub.

**Bước 1** — Tạo tài khoản tại [epicgames.com](https://www.epicgames.com)

**Bước 2** — Vào [unrealengine.com/en-US/ue-on-github](https://www.unrealengine.com/en-US/ue-on-github),
đăng nhập và chấp nhận EULA → link GitHub account.

**Bước 3** — Nhận email mời vào organization `EpicGames` trên GitHub.
Sau khi accept: `github.com/EpicGames/UnrealEngine` sẽ hiển thị repo (trước đó báo 404).

**Bước 4** — CARLA dùng **fork riêng của UE4**, không phải repo Epic gốc:
```
https://github.com/CarlaSimulator/UnrealEngine
```
Đây là UE4.26 đã patch thêm feature CARLA cần. Phải dùng fork này, không dùng vanilla UE4.26.

---

## 3. Clone và build Unreal Engine 4.26

> ⏱ Bước này tốn nhiều nhất: **4–8 giờ** build trên i5-9300H, ~50GB disk.

### 3.1 Clone UE4 (CARLA fork)

Mở Git Bash hoặc PowerShell với quyền Admin:

```bash
# Chọn ổ đĩa còn nhiều dung lượng nhất (ít nhất 150GB free)
cd D:\  # hoặc E:\ tuỳ máy

git clone --depth=1 -b carla https://github.com/CarlaSimulator/UnrealEngine.git
# --depth=1 bỏ full history → tiết kiệm ~10GB, clone nhanh hơn

cd UnrealEngine
```

> **Lưu ý**: `--depth=1` đủ để build. Nếu cần switch branch sau này thì
> `git fetch --unshallow`.

### 3.2 Chạy Setup

```batch
Setup.bat
```

Script này tải về các binary dependency của UE4 (~3–5GB). Có thể mất 30–60 phút
tùy tốc độ internet. Nếu bị ngắt giữa chừng, chạy lại — nó resume được.

### 3.3 Generate project files

```batch
GenerateProjectFiles.bat -2019
# Flag -2019 để generate cho Visual Studio 2019
```

Sinh ra `UE4.sln` trong thư mục gốc.

### 3.4 Build UE4 trong Visual Studio

Mở `UE4.sln` bằng Visual Studio 2019.

Trong Solution Explorer:
1. Click phải `UE4` → Set as StartUp Project
2. Chọn configuration: **Development Editor** | **Win64**
3. Menu Build → **Build Solution** (hoặc Ctrl+Shift+B)

> ⚠️ **RAM**: Bước này dùng đến 24–32GB RAM khi parallel compile. Xem §7 nếu
> máy chỉ có 16GB.

Sau khi build xong (~4–8h), kiểm tra:
```
UnrealEngine\Engine\Binaries\Win64\UE4Editor.exe  ← phải tồn tại
```

---

## 4. Clone và setup CARLA

### 4.1 Clone CARLA repo

```bash
cd D:\  # cùng ổ với UE4 để tránh đường dẫn dài

git clone https://github.com/carla-simulator/carla.git
cd carla
git checkout 0.9.14  # tag version khớp với build hiện tại của project
```

### 4.2 Set biến môi trường

```batch
# Thêm vào System Environment Variables (hoặc chạy trước mỗi build session)
setx UE4_ROOT "D:\UnrealEngine"
```

Verify:
```batch
echo %UE4_ROOT%
# Phải in ra: D:\UnrealEngine
```

### 4.3 Tải CARLA assets

```batch
# Trong thư mục carla\
Update.bat
```

Script này tải về content assets của CARLA (~20–30GB: xe, người, map mặc định,
prop, texture). Dùng aria2 nếu đã cài để tải nhanh hơn.

> Nếu `Update.bat` báo lỗi aria2 not found: cài `choco install aria2` rồi
> chạy lại, hoặc xoá flag aria2 trong script để dùng wget.

---

## 5. Build CARLA lần đầu

### 5.1 Build toàn bộ (PythonAPI + Editor)

```batch
# Trong thư mục carla\
make PythonAPI ARGS="--python-version=3.8"
make launch
```

- `make PythonAPI`: build `carla` Python package (`.egg` file) — cần thiết để
  Python client kết nối được CARLA server từ build mới.
- `make launch`: build CARLA plugin + mở UE4 Editor với CARLA project lần đầu
  (lần đầu build thêm ~1–2 giờ nữa do compile shaders).

### 5.2 Kiểm tra Editor mở được

Khi UE4 Editor mở thành công, bạn thấy:
- Content Browser với thư mục `Carla/`, `CarlaExporter/`, `Static/`, v.v.
- Viewport hiển thị Town01 hoặc map mặc định
- Menu `Carla` trên thanh menu UE4

---

## 6. Import custom 3D asset vào CARLA Editor

Đây là mục tiêu chính: thêm tòa nhà/xe tùy chỉnh vào build.

### 6.1 Chuẩn bị file 3D

CARLA Editor nhận file **FBX** (khuyến nghị) hoặc OBJ.

**Nguồn model:**
- Blender (miễn phí) → export FBX
- Sketchfab (có sẵn nhiều tòa nhà/xe, một số miễn phí)
- OpenStreetMap building footprint → extrude bằng Blender/Houdini → FBX

**Yêu cầu FBX trước khi import:**
- Scale: 1 unit = 1 cm (UE4 convention)
- Pivot point: tại gốc tọa độ hợp lý (gốc tòa nhà = mặt đất)
- Texture: đính kèm hoặc baked vào material
- Polygon count: ≤ 50k tris cho prop đơn (tòa nhà đơn giản ~5–15k tris là ổn)

### 6.2 Import vào UE4 Editor

1. Trong **Content Browser**: chuột phải → Import to /Game/...
2. Chọn file `.fbx`
3. Hộp thoại **FBX Import Options**:
   - ✅ Import Mesh
   - ✅ Generate Missing Collision (tự sinh collision box)
   - ✅ Import Textures (nếu có)
   - Mesh LOD: **0** (không cần LOD cho project nghiên cứu)
4. Click **Import All**

Asset xuất hiện trong Content Browser dưới dạng **Static Mesh**.

### 6.3 Tạo Blueprint actor (để spawn từ Python)

Để spawn asset từ Python script (`client.spawn_actor(...)`), cần đăng ký vào
CARLA blueprint library:

1. Content Browser: chuột phải Static Mesh → **Create Blueprint Class**
2. Đặt tên theo format CARLA: `BP_Building_MoLao_01`
3. Trong Blueprint Editor: thêm Static Mesh Component, gán mesh vừa import
4. Compile và Save

**Đăng ký vào CARLA's PropRegistry** (để hiện trong `world.get_blueprint_library()`):

Mở file `carla\Unreal\CarlaUE4\Config\CarlaSettings.ini`:
```ini
[CARLA/ObjectLabels]
; Thêm dòng này
BP_Building_MoLao_01=Building
```

Hoặc dùng `PropRegistry` trong `carla\Unreal\CarlaUE4\Plugins\Carla\Source\`:
```cpp
// PropRegistry.h — thêm entry
{"static.prop.building_molao_01", "BP_Building_MoLao_01"}
```

Sau đó từ Python:
```python
bp = world.get_blueprint_library().find("static.prop.building_molao_01")
transform = carla.Transform(carla.Location(x=100, y=50, z=0))
world.spawn_actor(bp, transform)
```

### 6.4 Thêm xe máy VN tùy chỉnh

Xe cần phức tạp hơn prop vì cần rig (bánh xe xoay, lái):

1. Import FBX xe → Static Mesh
2. Tạo **Skeletal Mesh** nếu cần animation bánh xe (optional)
3. Kế thừa Blueprint từ `BaseVehiclePawn` (class gốc của CARLA):
   - `BP_Vehicle_Motorbike_VN` extends `Carla/Blueprints/Vehicle/VehicleBP`
4. Đăng ký trong `VehicleFactory`:
   - File: `carla\Unreal\CarlaUE4\Plugins\Carla\Source\Carla\Vehicle\VehicleSpawnPoint.h`
   - Thêm entry tương tự các xe sẵn có

> **Tắt đường tắt**: Nếu chỉ cần prop trang trí (không cần physics xe thật),
> dùng Static Mesh Blueprint thay vì Vehicle BP — đơn giản hơn nhiều.

---

## 7. Workaround 16GB RAM

Build UE4 parallel compilation có thể dùng 24–32GB. Với 16GB cần giảm tải.

### 7.1 Tăng Virtual Memory (Pagefile)

```
Control Panel → System → Advanced System Settings 
→ Performance → Settings → Advanced → Virtual Memory → Change
```

Tắt "Automatically manage", chọn ổ SSD:
- Initial size: **16384 MB** (16GB)
- Maximum size: **32768 MB** (32GB)

> Dùng SSD (không phải HDD) — pagefile trên HDD quá chậm, build sẽ mất hàng
> chục giờ hoặc treo.

### 7.2 Giới hạn số luồng compile

Mở `Engine\Saved\UnrealBuildTool\BuildConfiguration.xml` (tạo mới nếu chưa có):

```xml
<?xml version="1.0" encoding="utf-8" ?>
<Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">
  <BuildConfiguration>
    <!-- Giới hạn 4 luồng thay vì mặc định = số core -->
    <MaxParallelActions>4</MaxParallelActions>
    <!-- Giảm memory per action -->
    <bAllowXGE>false</bAllowXGE>
  </BuildConfiguration>
</Configuration>
```

Với `MaxParallelActions=4`, RAM peak giảm từ ~28GB xuống ~14–18GB — có thể
vừa đủ với 16GB + 16GB pagefile.

### 7.3 Đóng mọi ứng dụng khác khi build

Tắt: trình duyệt, CARLA server, IDE, Discord, v.v. Giữ chỉ VS2019 và Task Manager.
Mục tiêu: để hệ thống có ~14GB RAM free trước khi bắt đầu build.

### 7.4 Chạy build qua đêm

Với `MaxParallelActions=4` trên i5-9300H 8T, build UE4 có thể mất 10–16 giờ
(thay vì 4–8h trên máy đủ RAM). Chạy trước khi ngủ:

```batch
@echo off
cd D:\UnrealEngine
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" ^
  UE4.sln ^
  /t:Build ^
  /p:Configuration="Development Editor" ^
  /p:Platform=Win64 ^
  /m:4 ^
  /flp:logfile=build_log.txt;verbosity=minimal
echo BUILD DONE >> build_log.txt
```

> Nếu build bị crash giữa chừng do OOM: tăng pagefile thêm, giảm
> `MaxParallelActions` xuống 2, chạy lại — MSBuild tiếp tục từ chỗ dừng (incremental build).

### 7.5 Rủi ro thực tế với 16GB

| Tình huống | Khả năng xảy ra |
|---|---|
| Build thành công nhưng chậm (10–16h) | Cao nếu pagefile đủ lớn trên SSD |
| Build crash lần đầu do OOM | Trung bình — giải quyết bằng giảm parallel |
| UE4 Editor mở nhưng lag khi làm việc | Cao — Editor cần 8–12GB RAM riêng |
| Shader compilation crash trong Editor | Trung bình — chạy lại sẽ tiếp tục |

---

## 8. Đóng gói thành packaged build

Sau khi import xong asset và test trong Editor, tạo build packaged mới:

### 8.1 Từ UE4 Editor (GUI)

```
File → Package Project → Windows → Windows (64-bit)
→ Chọn output folder (ví dụ: D:\CARLA_custom_build\)
```

Quá trình này: cook assets → compile shaders → package → ~1–2 giờ.

Output: thư mục `WindowsNoEditor\` với `CarlaUE4.exe` — thay thế build cũ của project.

### 8.2 Từ command line

```batch
cd D:\carla
make package
```

Output tại `carla\Dist\CARLA_0.9.14\` — có thể copy thay thế
`E:\School_project\finalproject\WindowsNoEditor\`.

### 8.3 Cờ quality level

Trong build mới, **bỏ cờ `-quality-level=Low`**:

```python
# start_carla.bat hoặc script khởi động
CarlaUE4.exe -carla-server -quality-level=Epic
# hoặc Medium nếu RTX 4060 8GB bị nặng khi chạy cùng AI pipeline
```

Với `Medium`/`Epic`, mesh xe + prop sẽ render được ở camera gần.

---

## 9. Tích hợp với pipeline hiện tại

Sau khi có build mới, pipeline hiện tại **không cần sửa** vì:
- `carla` Python package build lại từ source → API giữ nguyên
- `camera_controller.py`, `traffic_generator.py`, `scenario_controller.py` dùng
  Python API → tương thích

**Chỉ cần cập nhật:**

```python
# custom_tracking_system/config/camera_config.yaml
carla:
  host: localhost
  port: 2000
  quality_level: Medium  # thay vì Low
```

```python
# Map/buildings_render.py — có thể dùng spawn_actor thay debug.draw_box
# khi asset tòa nhà đã được import và đăng ký
bp = world.get_blueprint_library().find("static.prop.building_molao_01")
world.spawn_actor(bp, transform)
# → tòa nhà hiển thị được ở camera gần, không chỉ top-down nữa
```

**PythonAPI mới** (từ source build) cần copy vào venv:

```batch
# Sau make PythonAPI, copy egg file:
copy D:\carla\PythonAPI\carla\dist\carla-0.9.14-cp38-cp38-win_amd64.egg ^
     E:\School_project\finalproject\custom_tracking_system\venv_tracking\Lib\site-packages\
```

> Lưu ý: build từ source sinh egg cho **Python 3.8**. Venv hiện tại (`venv_tracking`)
> dùng Python 3.10. Cần tạo thêm venv 3.8 riêng, hoặc dùng wheel build thay vì egg.
> Kiểm tra version trước khi bắt đầu toàn bộ quá trình.

---

## 10. Troubleshooting

### Lỗi: `The command "Setup.bat" failed`

```
→ Kiểm tra GitHub account đã được add vào EpicGames org chưa
→ Chạy lại Setup.bat (nó retry tự động)
→ Nếu có proxy: set HTTP_PROXY/HTTPS_PROXY trước khi chạy
```

### Lỗi: `error C2589: '(' : illegal token on right side of '::'`

```
→ Windows SDK version sai. Cài thêm SDK 10.0.18362 trong VS2019 Installer.
→ Project Properties → General → Windows SDK Version → 10.0.18362.0
```

### Lỗi: `OutOfMemoryException` khi build

```
→ Giảm MaxParallelActions xuống 2
→ Tăng pagefile (§7.1)
→ Restart máy trước khi build (clear RAM fragmentation)
→ Build lại — incremental build tiếp tục từ chỗ dừng
```

### Lỗi: `make PythonAPI` báo Python version không khớp

```
→ Đảm bảo Python 3.8 trong PATH khi chạy make
→ Dùng: py -3.8 (Windows py launcher) thay vì python
→ Hoặc: set PYTHON_VERSION=3.8 trước make PythonAPI
```

### UE4 Editor crash khi mở CARLA project lần đầu

```
→ Shader compilation lần đầu cần ~8–16GB RAM. Đóng mọi app, chạy lại.
→ Nếu crash ở % nhất định: xoá thư mục carla\Unreal\CarlaUE4\Saved\,
  mở lại Editor → nó compile lại shader từ đầu.
```

### Mesh import xong nhưng không hiện trong viewport

```
→ Double click Static Mesh trong Content Browser → xem trong Mesh Viewer
→ Nếu mesh đen: thiếu material. Tạo material mới, assign.
→ Nếu mesh không thấy khi spawn: kiểm tra collision (auto-collision
  có thể block visibility). Trong Detail panel: Collision → No Collision.
```

---

## Tóm tắt thời gian ước tính (i5-9300H, 16GB RAM + 16GB pagefile SSD)

| Bước | Thời gian |
|---|---|
| Cài prerequisites | 1–2h |
| Clone UE4 + Setup.bat | 1–2h (tùy internet) |
| Build UE4 (Development Editor) | **10–16h** (qua đêm) |
| Clone CARLA + Update.bat (assets) | 2–4h |
| Build PythonAPI + make launch | 2–3h |
| Compile shaders lần đầu trong Editor | 1–2h |
| Import asset + đăng ký Blueprint | 1–2h/loại asset |
| Package build mới | 1–2h |
| **Tổng** | **~20–33h** (1–2 ngày làm việc) |

---

*Tài liệu tạo: 2026-06-17. Áp dụng cho CARLA 0.9.14, UE4.26 (CarlaSimulator fork),
Windows 10/11, Visual Studio 2019.*
