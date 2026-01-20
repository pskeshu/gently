@echo off
REM ===================================================================
REM Gently Services Launcher
REM ===================================================================
REM
REM This script starts all required services for the Gently system:
REM   1. Simple Server (Microscope API on port 60610)
REM   2. SAM Server (Segmentation model on port 18862)
REM
REM Note: Visualization Server is now started by the copilot automatically.
REM Note: Perception system runs within the copilot (no separate service).
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
echo GENTLY SERVICES LAUNCHER
echo ======================================================================
echo.

REM Check if Micro-Manager server is running
echo [1/2] Checking Micro-Manager connection...
call "%VENV_PATH%\Scripts\activate.bat"
cd /d %~dp0
python -c "from client import get_mmc; core = get_mmc(); print('     OK - Micro-Manager connected')" 2>nul
if errorlevel 1 (
    echo     WARNING: Cannot connect to Micro-Manager
    echo     Make sure Micro-Manager is running before starting services
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)
echo.

REM Start all services
echo [2/2] Starting services...
echo.

REM Start Simple Server (Microscope API) in a new window
echo Starting Simple Microscope Server...
start "Simple Microscope Server" cmd /k "cd /d %~dp0 && call %VENV_PATH%\Scripts\activate.bat && python backend/simple_server.py"

REM Give server a moment to start
timeout /t 3 /nobreak > nul

REM Start SAM server in a new window
echo Starting SAM Server...
start "SAM Server" cmd /k "cd /d %~dp0 && call %VENV_PATH%\Scripts\activate.bat && python backend/sam_server.py"

echo.
echo All services started!
echo.
echo ======================================================================
echo Services running:
echo   - Microscope API:        http://127.0.0.1:60610
echo   - SAM Server:            localhost:18862
echo.
echo Note: Visualization Server starts with the copilot (port 8080)
echo Note: Perception runs within the copilot (no separate service)
echo ======================================================================
echo.
echo Close the service windows to stop individual services.
echo.
pause
