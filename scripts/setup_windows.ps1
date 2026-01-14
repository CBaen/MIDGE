# ============================================================
# MIDGE Windows Setup Script
# ============================================================
#
# This script:
#   1. Installs Python dependencies
#   2. Sets up Qdrant collections
#   3. Creates Windows Task Scheduler task for auto-start
#
# Run as Administrator:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$MidgePath = "C:\Users\baenb\projects\MIDGE"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MIDGE Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to MIDGE directory
Set-Location $MidgePath

# Step 1: Install Python dependencies
Write-Host "[1/4] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Done!" -ForegroundColor Green

# Step 2: Check Docker/Qdrant
Write-Host ""
Write-Host "[2/4] Checking Docker and Qdrant..." -ForegroundColor Yellow

$dockerRunning = $false
try {
    docker info 2>&1 | Out-Null
    $dockerRunning = $true
    Write-Host "  Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Docker not running. Please start Docker Desktop." -ForegroundColor Yellow
}

if ($dockerRunning) {
    $qdrantRunning = docker ps | Select-String "qdrant"
    if (-not $qdrantRunning) {
        Write-Host "  Starting Qdrant container..." -ForegroundColor Yellow
        $existing = docker ps -a | Select-String "qdrant"
        if ($existing) {
            docker start qdrant
        } else {
            docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
        }
        Start-Sleep -Seconds 5
    }
    Write-Host "  Qdrant is running" -ForegroundColor Green
}

# Step 3: Setup Qdrant collections
Write-Host ""
Write-Host "[3/4] Setting up Qdrant collections..." -ForegroundColor Yellow
python scripts/setup_qdrant.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Qdrant setup had issues. Check if Qdrant is running." -ForegroundColor Yellow
} else {
    Write-Host "  Done!" -ForegroundColor Green
}

# Step 4: Create Task Scheduler task
Write-Host ""
Write-Host "[4/4] Creating Windows Task Scheduler task..." -ForegroundColor Yellow

$taskName = "MIDGE_AutoStart"
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "  Task already exists. Updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$MidgePath\scripts\start_midge.bat`"" -WorkingDirectory $MidgePath
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Start MIDGE trading intelligence on Windows startup"
    Write-Host "  Task created: $taskName" -ForegroundColor Green
    Write-Host "  MIDGE will start automatically when Windows boots." -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Could not create scheduled task. Run as Administrator." -ForegroundColor Yellow
    Write-Host "  You can manually add start_midge.bat to your Startup folder." -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start MIDGE now:"
Write-Host "  python run.py" -ForegroundColor White
Write-Host ""
Write-Host "To start individual services:"
Write-Host "  python run.py --evolution-only     # Just the learning loop" -ForegroundColor White
Write-Host "  python run.py --ingest-only        # Just data collection" -ForegroundColor White
Write-Host "  python run.py --dashboard-only     # Just the dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Dashboard will be at: http://localhost:8080"
Write-Host ""
