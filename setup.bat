@echo off
setlocal EnableDelayedExpansion
title Demo Setup (chay 1 lan)

REM ================================================================
REM  setup.bat  --  Chuan bi 1 lan de sau do chay demo bang 1 lenh.
REM
REM  Lam 3 viec:
REM    1) Build frontend (Vite -> frontend/dist, tinh, khong can Node khi demo)
REM    2) Copy video CCTV that (Pho Hue) sang path ASCII data/demo_pho_hue.mp4
REM    3) Kiem tra model weights + venv co san chua
REM
REM  Sau khi chay xong setup.bat 1 lan, moi lan demo chi can: run_demo.bat
REM ================================================================

set ROOT=%~dp0
set PY=%ROOT%custom_tracking_system\venv_tracking\Scripts\python.exe

echo.
echo  ============================================
echo   SETUP DEMO (chay 1 lan)
echo  ============================================
echo.

REM --- Kiem tra venv Python ---
if not exist "%PY%" (
    echo [ERROR] Khong tim thay venv:
    echo         %PY%
    echo Tao venv va cai dependencies truoc ^(xem docs/chay_demo_video_cctv.md^).
    pause & exit /b 1
)

REM ================================================================
REM  BUOC 1: Build frontend
REM ================================================================
echo [1/3] Build frontend ^(Vite^)...
cd /d "%ROOT%frontend"
if not exist node_modules (
    echo       Cai node_modules ^(lan dau^)...
    call npm install
    if errorlevel 1 ( echo [ERROR] npm install that bai. Da cai Node.js chua? & pause & exit /b 1 )
)
call npm run build
if errorlevel 1 ( echo [ERROR] npm run build that bai. & pause & exit /b 1 )
cd /d "%ROOT%"
echo       frontend/dist da san sang.
echo.

REM ================================================================
REM  BUOC 2: Chuan bi video demo (path ASCII)
REM ================================================================
echo [2/3] Chuan bi video demo...
"%PY%" "%ROOT%scripts\prepare_demo_video.py"
if errorlevel 1 ( echo [WARN] Chua co video demo — xem thong bao o tren. )
echo.

REM ================================================================
REM  BUOC 3: Kiem tra weights
REM ================================================================
echo [3/3] Kiem tra model weights...
set MISS=0
if not exist "%ROOT%custom_tracking_system\weights\yolo11s_vn.pt" ( echo [WARN] thieu yolo11s_vn.pt & set MISS=1 )
if not exist "%ROOT%custom_tracking_system\weights\goal_classifier_real.pkl" ( echo [WARN] thieu goal_classifier_real.pkl & set MISS=1 )
if !MISS!==0 echo       weights day du.
echo.

echo  ============================================
echo   SETUP XONG. Chay demo bang:  run_demo.bat
echo  ============================================
echo.
pause
