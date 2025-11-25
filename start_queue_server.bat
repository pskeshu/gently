@echo off
REM ===================================================================
REM Bluesky Queue Server Launcher
REM ===================================================================
REM
REM This script starts the Bluesky Queue Server components:
REM   1. RE Manager (RunEngine manager with ZMQ sockets)
REM   2. HTTP Server (REST API on port 60610)
REM
REM The Queue Server will use the configuration and devices defined in:
REM   - backend/queue_server_config.yml
REM   - backend/queue_server_startup.py
REM
REM ===================================================================

echo.
echo ======================================================================
echo BLUESKY QUEUE SERVER LAUNCHER
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
    echo     Make sure Micro-Manager is running before starting Queue Server
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)
echo.

REM Start Queue Server components
echo [3/3] Starting Bluesky Queue Server...
echo.
echo The Queue Server will start with:
echo   - Control socket: tcp://127.0.0.1:60615
echo   - Document stream: tcp://127.0.0.1:60645
echo   - HTTP REST API: http://127.0.0.1:60610
echo.
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

REM Start HTTP server first in a new window
echo Starting HTTP Server in new window...
start "Bluesky HTTP Server" cmd /k "call venv\Scripts\activate.bat && set "QSERVER_ZMQ_CONTROL_ADDRESS=tcp://127.0.0.1:60615" && start-bluesky-httpserver --host 127.0.0.1 --port 60610 --public"

REM Give HTTP server a moment to start
timeout /t 2 /nobreak > nul

REM Start RE Manager in this window (keeps running)
REM The --startup-script flag points to our device initialization
echo.
echo Starting RE Manager (this window will show RE Manager output)...
echo.
start-re-manager ^
    --startup-script=backend/queue_server_startup.py ^
    --zmq-publish-console=ON ^
    --zmq-control-addr=tcp://127.0.0.1:60615 ^
    --zmq-info-addr=tcp://127.0.0.1:60625 ^
    --redis-addr=
