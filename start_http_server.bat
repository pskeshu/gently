@echo off
REM ===================================================================
REM Bluesky HTTP Server Launcher
REM ===================================================================
REM
REM This script starts the HTTP REST API server for the Queue Server.
REM It connects to the RE Manager via ZMQ sockets.
REM
REM IMPORTANT: start_queue_server.bat must be running first!
REM
REM ===================================================================

echo.
echo ======================================================================
echo BLUESKY HTTP SERVER LAUNCHER
echo ======================================================================
echo.

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo     OK - Virtual environment activated
echo.

REM Start HTTP Server
echo [2/2] Starting HTTP REST API server...
echo.
echo The HTTP server will be available at:
echo   http://127.0.0.1:60610
echo.
echo API Documentation:
echo   http://127.0.0.1:60610/docs
echo.
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

bluesky-httpserver ^
    --zmq-control-addr=tcp://127.0.0.1:60615 ^
    --server-host=127.0.0.1 ^
    --server-port=60610

pause
