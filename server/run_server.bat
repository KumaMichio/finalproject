@echo off
REM ============================================================
REM  Run Backend API Server
REM ============================================================
REM
REM  Cach 1: Chi chay API server (khong can CARLA)
REM    run_server.bat
REM
REM  Cach 2: Chay API server + AI pipeline (can CARLA dang chay)
REM    run_server.bat --with-ai
REM
REM ============================================================

cd /d "%~dp0"

echo ============================================
echo  Multi-Camera CCTV Tracking System - Server
echo ============================================

REM CARLA Python API path (can chinh sua neu CARLA cai o noi khac)
set CARLA_ROOT=%~dp0..\WindowsNoEditor
set CARLA_PYTHON=%CARLA_ROOT%\PythonAPI\carla\dist\carla_extracted
set CARLA_API=%CARLA_ROOT%\PythonAPI
set PYTHONPATH=%CARLA_PYTHON%;%CARLA_API%;%PYTHONPATH%

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

REM Chay server
echo Starting server...
echo.

if "%1"=="--with-ai" (
    echo Mode: API Server + AI Pipeline (CARLA required on localhost:2000)
    echo.
    python app.py --with-ai --host 0.0.0.0 --port 8000
) else (
    echo Mode: API Server only (no CARLA needed)
    echo.
    python app.py --host 0.0.0.0 --port 8000
)

pause
