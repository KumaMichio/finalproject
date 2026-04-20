@echo off
REM Multi-Camera CCTV Tracking System Launcher
REM Usage: run.bat [max_frames] [config_file]

cd /d %~dp0

echo Starting Multi-Camera CCTV Tracking System...
echo.

REM === CARLA Python API path ===
set CARLA_EXTRACTED=e:\finalproject\WindowsNoEditor\PythonAPI\carla\dist\carla_extracted
set CARLA_PYTHONAPI=e:\finalproject\WindowsNoEditor\PythonAPI

if not exist "%CARLA_EXTRACTED%\carla\libcarla.cp37-win_amd64.pyd" (
    echo ERROR: Extracted CARLA package not found.
    echo Expected: %CARLA_EXTRACTED%\carla\libcarla.cp37-win_amd64.pyd
    echo See execute.md - Section 2 for setup instructions.
    pause
    exit /b 1
)

set PYTHONPATH=%CARLA_EXTRACTED%;%CARLA_PYTHONAPI%;%PYTHONPATH%

REM === Default run arguments ===
set MAX_FRAMES=%1
if "%MAX_FRAMES%"=="" set MAX_FRAMES=1000

set CONFIG_FILE=%2
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config\camera_config.yaml

echo Configuration:
echo   Max Frames: %MAX_FRAMES%
echo   Config:     %CONFIG_FILE%
echo   CARLA egg:  %CARLA_EGG%
echo.
echo (Make sure CarlaUE4.exe is already running)
echo.

python main.py --config %CONFIG_FILE% --max-frames %MAX_FRAMES% --log-level INFO

echo.
echo Tracking system finished.
pause