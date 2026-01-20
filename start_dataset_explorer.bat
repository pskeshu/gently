@echo off
REM Start the Gently Dataset Explorer Web UI
REM Browse sessions, embryos, images and manage ground truth annotations

echo Starting Gently Dataset Explorer...
echo.
echo Web UI will be available at: http://localhost:8765
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python -m gently.dataset.cli serve --port 8765
