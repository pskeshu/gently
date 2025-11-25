@echo off
REM ===================================================================
REM Simple Microscope Server Launcher
REM ===================================================================
REM
REM This script starts the simple microscope server that runs
REM RunEngine in the main thread, avoiding Windows threading issues.
REM
REM Usage:
REM   1. Start this server: start_simple_server.bat
REM   2. Start SAM server: python backend/sam_server.py
REM   3. Run agent: python run_microscope_agent.py
REM
REM ===================================================================

echo.
echo ======================================================================
echo SIMPLE MICROSCOPE SERVER
echo ======================================================================
echo.

REM Activate virtual environment
echo [1/2] Activating virtual environment...
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
echo [2/2] Checking Micro-Manager connection...
python -c "from client import get_mmc; core = get_mmc(); print('     OK - Micro-Manager connected')" 2>nul
if errorlevel 1 (
    echo     WARNING: Cannot connect to Micro-Manager
    echo     Make sure Micro-Manager is running with the server plugin
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)
echo.

REM Start the simple server
echo Starting Simple Microscope Server...
echo.
echo The server will start with:
echo   - HTTP API: http://127.0.0.1:60610
echo   - RunEngine running in main thread
echo.
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

python backend/simple_server.py
