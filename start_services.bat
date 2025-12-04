@echo off
REM ===================================================================
REM Gently Services Launcher
REM ===================================================================
REM
REM This script starts all required services for the Gently system:
REM   1. Simple Server (Microscope API on port 60610)
REM   2. SAM Server (Segmentation model on port 18862)
REM   3. CV Subagent (Computer Vision analysis on port 8100)
REM
REM Note: Visualization Server is now started by the copilot automatically.
REM
REM Run this before launching the copilot.
REM
REM ===================================================================

echo.
echo ======================================================================
echo GENTLY SERVICES LAUNCHER
echo ======================================================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please ensure venv exists: python -m venv venv
    pause
    exit /b 1
)
echo     OK - Virtual environment activated
echo.

REM Check if Micro-Manager server is running
echo [2/3] Checking Micro-Manager connection...
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
echo [3/3] Starting services...
echo.

REM Start Simple Server (Microscope API) in a new window
echo Starting Simple Microscope Server...
start "Simple Microscope Server" cmd /k "call venv\Scripts\activate.bat && python backend/simple_server.py"

REM Give server a moment to start
timeout /t 3 /nobreak > nul

REM Start SAM server in a new window
echo Starting SAM Server...
start "SAM Server" cmd /k "call venv\Scripts\activate.bat && python backend/sam_server.py"

REM Give SAM server a moment to start
timeout /t 2 /nobreak > nul

REM Start CV Subagent in a new window
echo Starting CV Subagent...
start "CV Subagent" cmd /k "call venv\Scripts\activate.bat && python start_cv_service.py"

echo.
echo All services started!
echo.
echo ======================================================================
echo Services running:
echo   - Microscope API:        http://127.0.0.1:60610
echo   - SAM Server:            localhost:18862
echo   - CV Subagent:           http://localhost:8100
echo.
echo Note: Visualization Server starts with the copilot (port 8080)
echo ======================================================================
echo.
echo Close the service windows to stop individual services.
echo.
pause
