@echo off
REM ===================================================================
REM Gently Services Launcher
REM ===================================================================
REM
REM This script starts the Gently Device Layer - a unified server that
REM provides all hardware control and SAM detection in a single process.
REM
REM The device layer replaces the previous 3-process architecture:
REM   - MMCore RPyC server (eliminated - now direct init)
REM   - simple_server.py HTTP (replaced by device_layer.py)
REM   - sam_server.py RPyC (replaced by HTTP endpoints in device_layer.py)
REM
REM Run this before launching the copilot.
REM
REM ===================================================================

REM Determine the venv path - check local first, then main repo
set "VENV_PATH=%~dp0venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    set "VENV_PATH=C:\Users\dispim\Documents\GitHub\gently\venv"
)
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Cannot find virtual environment
    echo Looked in: %~dp0venv
    echo Looked in: C:\Users\dispim\Documents\GitHub\gently\venv
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo GENTLY DEVICE LAYER LAUNCHER
echo ======================================================================
echo.

REM Start Device Layer in a new window
echo Starting Gently Device Layer...
echo   - Direct MMCore initialization (no external Micro-Manager needed)
echo   - HTTP API on port 60610
echo   - SAM detection via /api/detect_embryos
echo.

start "Gently Device Layer" cmd /k "cd /d %~dp0 && call %VENV_PATH%\Scripts\activate.bat && python start_device_layer.py"

echo.
echo Device Layer starting!
echo.
echo ======================================================================
echo Services:
echo   - Device Layer API:  http://127.0.0.1:60610
echo   - SAM Detection:     http://127.0.0.1:60610/api/detect_embryos
echo.
echo Note: Visualization Server starts with the copilot (port 8080)
echo ======================================================================
echo.
echo Close the Device Layer window to stop the service.
echo.
pause
