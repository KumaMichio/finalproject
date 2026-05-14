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

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Chay server
echo Starting server...
echo.

if "%1"=="--with-ai" (
    echo Mode: API Server + AI Pipeline (CARLA required)
    echo Make sure CARLA server is running on localhost:2000
    echo.
    python app.py --with-ai --host 0.0.0.0 --port 8000
) else (
    echo Mode: API Server only (no CARLA needed)
    echo.
    python app.py --host 0.0.0.0 --port 8000
)

pause
