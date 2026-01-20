@echo off
REM ===================================================================
REM Gently Services Restarter
REM ===================================================================
REM Stops all services and restarts them.
REM ===================================================================

echo.
echo ======================================================================
echo RESTARTING GENTLY SERVICES
echo ======================================================================
echo.

REM Stop existing services
call "%~dp0stop_services.bat"

REM Wait a moment for ports to be released
timeout /t 2 /nobreak > nul

REM Start services again
call "%~dp0start_services.bat"
