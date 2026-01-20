# ===================================================================
# Gently Services Launcher (PowerShell)
# ===================================================================
#
# This script starts all required services for the Gently system:
#   1. Simple Server (Microscope API on port 60610)
#   2. SAM Server (Segmentation model on port 18862)
#
# Note: Visualization Server is now started by the copilot automatically.
# Note: Perception system runs within the copilot (no separate service).
#
# Run this before launching the copilot.
#
# Usage:
#   .\start_services.ps1           # Start all services
#   .\start_services.ps1 -NoSAM    # Skip SAM server
#
# ===================================================================

param(
    [switch]$NoSAM,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Header($text) {
    Write-Host "`n$text" -ForegroundColor Cyan
}

function Write-Success($text) {
    Write-Host "  OK - $text" -ForegroundColor Green
}

function Write-Warning($text) {
    Write-Host "  WARNING: $text" -ForegroundColor Yellow
}

function Write-Error($text) {
    Write-Host "  ERROR: $text" -ForegroundColor Red
}

function Write-Info($text) {
    Write-Host "  $text" -ForegroundColor Gray
}

# Banner
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "                    GENTLY SERVICES LAUNCHER" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Step 1: Check/Activate virtual environment
Write-Header "[1/3] Checking virtual environment..."

$VenvPath = Join-Path $ScriptDir "venv"
$VenvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"

if (Test-Path $VenvActivate) {
    & $VenvActivate
    Write-Success "Virtual environment activated"
} else {
    Write-Error "Virtual environment not found at: $VenvPath"
    Write-Info "Create it with: python -m venv venv"
    if (-not $NoPause) { Read-Host "Press Enter to exit" }
    exit 1
}

# Step 2: Check Micro-Manager connection
Write-Header "[2/3] Checking Micro-Manager connection..."

try {
    $result = python -c "from client import get_mmc; core = get_mmc(); print('connected')" 2>&1
    if ($result -match "connected") {
        Write-Success "Micro-Manager connected"
    } else {
        throw "Connection failed"
    }
} catch {
    Write-Warning "Cannot connect to Micro-Manager"
    Write-Info "Make sure Micro-Manager is running with rpyc server enabled"

    $response = Read-Host "Continue anyway? (Y/N)"
    if ($response -notmatch "^[Yy]") {
        exit 1
    }
}

# Step 3: Start services
Write-Header "[3/3] Starting services..."
Write-Host ""

$services = @()

# Start Simple Server
Write-Host "  Starting Simple Microscope Server..." -ForegroundColor White
$simpleServer = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ScriptDir'; & '$VenvActivate'; python backend/simple_server.py"
) -PassThru
$services += @{Name="Simple Server"; Process=$simpleServer; Port=60610}
Write-Success "Simple Server started (PID: $($simpleServer.Id))"

Start-Sleep -Seconds 3

# Start SAM Server (optional)
if (-not $NoSAM) {
    Write-Host "  Starting SAM Server..." -ForegroundColor White
    $samServer = Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$ScriptDir'; & '$VenvActivate'; python backend/sam_server.py"
    ) -PassThru
    $services += @{Name="SAM Server"; Process=$samServer; Port=18862}
    Write-Success "SAM Server started (PID: $($samServer.Id))"
} else {
    Write-Info "Skipping SAM Server (-NoSAM flag)"
}

# Services Summary
Write-Header "Services Summary"
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Services running:" -ForegroundColor White
Write-Host ""

foreach ($svc in $services) {
    $status = if ($svc.Process.HasExited) { "STOPPED" } else { "RUNNING" }
    $color = if ($status -eq "RUNNING") { "Green" } else { "Red" }
    Write-Host "    $($svc.Name.PadRight(20)) " -NoNewline
    Write-Host "[$status]" -ForegroundColor $color -NoNewline
    Write-Host "  Port: $($svc.Port)  PID: $($svc.Process.Id)"
}

Write-Host ""
Write-Host "  Endpoints:" -ForegroundColor White
Write-Host "    Microscope API:     http://127.0.0.1:60610" -ForegroundColor Gray
if (-not $NoSAM) {
    Write-Host "    SAM Server:         localhost:18862 (rpyc)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "  Note: Visualization Server starts with copilot (port 8080)" -ForegroundColor DarkGray
Write-Host "  Note: Perception runs within the copilot (no separate service)" -ForegroundColor DarkGray
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Close the service windows to stop individual services." -ForegroundColor Yellow
Write-Host "Or run: Get-Process -Id $($services.Process.Id -join ',') | Stop-Process" -ForegroundColor DarkGray
Write-Host ""

if (-not $NoPause) {
    Read-Host "Press Enter to exit (services will keep running)"
}
