@echo off
echo Dung tat ca services...

REM Dung Frontend (Vite, port 3000)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /C:":3000 " ^| findstr "LISTENING"') do (
    echo Dung Frontend (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Dung Backend Server (port 8000)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /C:":8000 " ^| findstr "LISTENING"') do (
    echo Dung Backend Server (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Dung CARLA (CarlaUE4-Win64-Shipping.exe)
taskkill /IM "CarlaUE4-Win64-Shipping.exe" /F >nul 2>&1
if %errorlevel%==0 ( echo Dung CARLA. ) else ( echo CARLA khong chay. )

echo.
echo Xong.
pause
