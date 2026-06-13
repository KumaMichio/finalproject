@echo off
setlocal EnableDelayedExpansion
title CCTV System Launcher

REM ================================================================
REM  start.bat  --  Khoi dong toan bo he thong trong 1 lenh
REM
REM  Usage:
REM    start.bat           -- CARLA + Server + Frontend (full)
REM    start.bat --no-ai   -- Chi Server + Frontend (khong can CARLA)
REM ================================================================

set ROOT=%~dp0

REM ----------------------------------------------------------------
REM  CAU HINH (chinh sua khi can)
REM ----------------------------------------------------------------
set CARLA_EXE=%ROOT%WindowsNoEditor\CarlaUE4.exe
set SERVER_DIR=%ROOT%server
set FRONTEND_DIR=%ROOT%frontend

REM  Python dung de chay server (--with-ai can boxmot/torch/ultralytics/torchreid
REM  + goi `carla` pip — KHONG dung python mac dinh trong PATH, no thieu cac
REM  dependency nay). venv_tracking da co carla==0.9.15 (pip) ket noi duoc
REM  CARLA server 0.9.14 (chi canh bao version mismatch, van hoat dong).
set SERVER_PYTHON=%ROOT%custom_tracking_system\venv_tracking\Scripts\python.exe

REM  Them --half neu dung CARLA + AI cung luc (4GB VRAM, GTX 1050Ti)
set SERVER_FLAGS=--with-ai
REM  Doi thanh: set SERVER_FLAGS=--with-ai --half   (neu can FP16)
REM ----------------------------------------------------------------

REM --- Kiem tra mode ---
set NO_AI=0
if "%1"=="--no-ai" set NO_AI=1

echo.
echo  ============================================
echo   Multi-Camera CCTV Tracking System
echo  ============================================
echo.

REM ================================================================
REM  BUOC 0: Kiem tra frontend deps
REM ================================================================
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [SETUP] Chua co node_modules, dang chay npm install...
    cd /d "%FRONTEND_DIR%"
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install that bai. Kiem tra Node.js da cai chua.
        pause & exit /b 1
    )
    cd /d "%ROOT%"
    echo.
)

REM ================================================================
REM  BUOC 1: Khoi dong CARLA (bo qua neu --no-ai)
REM ================================================================
if %NO_AI%==1 goto skip_carla

if not exist "%CARLA_EXE%" (
    echo [ERROR] Khong tim thay CARLA tai:
    echo         %CARLA_EXE%
    echo.
    echo Chinh sua bien CARLA_EXE o dau file start.bat, hoac chay:
    echo   start.bat --no-ai    ^(khong can CARLA^)
    echo.
    pause & exit /b 1
)

echo [1/3] Khoi dong CARLA Simulator...
start "CARLA Simulator" "%CARLA_EXE%" -windowed -ResX=800 -ResY=600 -quality-level=Low
echo       Cho CARLA san sang tren port 2000...

set /a attempt=0
:wait_carla
timeout /t 3 /nobreak >nul
netstat -an 2>nul | findstr /C:":2000 " | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo       CARLA san sang ^(port 2000 OK^).
    goto carla_ok
)
set /a attempt+=1
if !attempt! lss 14 ( echo       Thu !attempt!/14 ... & goto wait_carla )
echo [WARN] Khong detect duoc port 2000, tiep tuc sau 14 lan thu.

:carla_ok
echo.

:skip_carla

REM ================================================================
REM  BUOC 2: Khoi dong Backend Server
REM ================================================================
if %NO_AI%==1 (
    set SERVER_FLAGS=
    echo [1/2] Khoi dong Backend Server ^(API only, khong co AI^)...
) else (
    echo [2/3] Khoi dong Backend Server ^(AI pipeline se ket noi CARLA^)...
)

start "Backend Server" cmd /k "cd /d "%SERVER_DIR%" && "%SERVER_PYTHON%" app.py %SERVER_FLAGS%"

echo       Cho server san sang tren port 8000...
set /a attempt=0
:wait_server
timeout /t 3 /nobreak >nul
netstat -an 2>nul | findstr /C:":8000 " | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo       Server san sang ^(port 8000 OK^).
    goto server_ok
)
set /a attempt+=1
if !attempt! lss 10 ( echo       Thu !attempt!/10 ... & goto wait_server )
echo [WARN] Khong detect duoc port 8000, tiep tuc sau 10 lan thu.

:server_ok
echo.

REM ================================================================
REM  BUOC 3: Khoi dong Frontend (Vite dev server)
REM ================================================================
if %NO_AI%==1 (
    echo [2/2] Khoi dong Web Dashboard...
) else (
    echo [3/3] Khoi dong Web Dashboard...
)
start "Frontend Dev" cmd /k "cd /d "%FRONTEND_DIR%" && npm run dev"

echo.
echo  ============================================
echo   He thong da khoi dong:
if %NO_AI%==0 echo   - CARLA Simulator  : chay nen ^(cua so rieng^)
echo   - Backend API      : http://localhost:8000
echo   - Swagger Docs     : http://localhost:8000/docs
echo   - Web Dashboard    : http://localhost:3000
echo  ============================================
echo.
echo  De dung he thong: dong cac cua so CARLA, Backend, Frontend.
echo  Nhan phim bat ky de dong launcher nay...
pause >nul
