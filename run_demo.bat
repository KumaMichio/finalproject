@echo off
setlocal EnableDelayedExpansion
title CCTV Demo (video that - 1 lenh)

REM ================================================================
REM  run_demo.bat  --  Chay TOAN BO demo bang 1 lenh.
REM
REM  - Nguon: video CCTV that (Pho Hue - Tran Khat Chan), KHONG can CARLA.
REM  - Backend FastAPI + AI pipeline chay o :8000, dong thoi serve luon
REM    frontend da build (SPA) -> 1 port, 1 tien trinh, khong can Vite.
REM  - Tu dong mo trinh duyet vao Dashboard khi server san sang.
REM
REM  Lan dau tien phai chay setup.bat truoc (build frontend + chuan bi video).
REM  Dung demo: dong cua so nay (hoac Ctrl+C).
REM ================================================================

set ROOT=%~dp0
set PY=%ROOT%custom_tracking_system\venv_tracking\Scripts\python.exe
set CONFIG=%ROOT%custom_tracking_system\config\camera_config_pho_hue.yaml

REM Video nguon: truyen duong dan lam tham so 1 de doi clip, vd:
REM    run_demo.bat "D:\clip_khac.mp4"
REM Khong truyen -> dung video demo mac dinh (Pho Hue) do setup.bat stage.
REM LUU Y: lan/den/vach + ban do dang gan cung cho view Pho Hue (CAM_PHO_HUE_REAL).
REM   - Clip KHAC cua CUNG camera Pho Hue -> chay dung het (toa do van khop).
REM   - Video dia diem KHAC -> detect/track/du doan huong van chay, nhung vi pham
REM     lan/den va ban do se KHONG dung (can config calibration/ROI rieng).
if not "%~1"=="" (
    set "VIDEO=%~1"
) else (
    set "VIDEO=%ROOT%data\demo_pho_hue.mp4"
)

echo.
echo  ============================================
echo   CCTV DEMO — video that (Pho Hue - Tran Khat Chan)
echo  ============================================
echo.

REM --- Kiem tra dieu kien ---
if not exist "%PY%" (
    echo [ERROR] Khong tim thay venv Python:
    echo         %PY%
    pause & exit /b 1
)
if not exist "%ROOT%frontend\dist\index.html" (
    echo [ERROR] Chua build frontend ^(frontend\dist^).
    echo         Chay:  setup.bat   truoc.
    pause & exit /b 1
)
if not exist "%VIDEO%" (
    echo [ERROR] Khong tim thay video: %VIDEO%
    echo         - Neu dung mac dinh: chay setup.bat truoc de stage video demo.
    echo         - Neu tu truyen: kiem tra lai duong dan ^(bo trong dau ngoac kep^).
    pause & exit /b 1
)
if not exist "%CONFIG%" (
    echo [ERROR] Khong tim thay config: %CONFIG%
    pause & exit /b 1
)

REM --- Mo trinh duyet khi server san sang (chay nen) ---
start "" "%PY%" "%ROOT%scripts\open_when_ready.py" 8000

echo  Dang khoi dong backend + AI pipeline tren video that...
echo  Dashboard se tu mo tai http://localhost:8000
echo  ^(Bao cao: tren video nay headline vi pham = "di sai lan"^)
echo.

REM --- Chay backend foreground (CWD = server de khop resolve config/graph) ---
cd /d "%ROOT%server"
"%PY%" app.py ^
  --with-ai ^
  --source file ^
  --video-path "%VIDEO%" ^
  --camera-ids CAM_PHO_HUE_REAL ^
  --config "%CONFIG%" ^
  --loop-video

echo.
echo  [DUNG] Backend da dung.
pause
