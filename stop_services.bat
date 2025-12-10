@echo off
REM ===================================================================
REM Gently Services Stopper
REM ===================================================================
REM Kills all running Gently services by finding Python processes
REM running the specific scripts.
REM ===================================================================

echo.
echo Stopping Gently services...
echo.

REM Kill processes by window title (matches the "start" command titles)
taskkill /FI "WINDOWTITLE eq Simple Microscope Server*" /F 2>nul
taskkill /FI "WINDOWTITLE eq SAM Server*" /F 2>nul
taskkill /FI "WINDOWTITLE eq CV Subagent*" /F 2>nul

REM Also kill by port in case window titles don't match
REM Find and kill process on port 60610 (Simple Server)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :60610 ^| findstr LISTENING 2^>nul') do (
    echo Killing process on port 60610 (PID: %%a)
    taskkill /PID %%a /F 2>nul
)

REM Find and kill process on port 18862 (SAM Server)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :18862 ^| findstr LISTENING 2^>nul') do (
    echo Killing process on port 18862 (PID: %%a)
    taskkill /PID %%a /F 2>nul
)

REM Find and kill process on port 8100 (CV Subagent)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8100 ^| findstr LISTENING 2^>nul') do (
    echo Killing process on port 8100 (PID: %%a)
    taskkill /PID %%a /F 2>nul
)

REM Find and kill process on port 8080 (Viz Server)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING 2^>nul') do (
    echo Killing process on port 8080 (PID: %%a)
    taskkill /PID %%a /F 2>nul
)

echo.
echo All services stopped.
echo.
