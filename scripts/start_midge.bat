@echo off
REM ============================================================
REM MIDGE Startup Script for Windows
REM ============================================================
REM
REM This script starts all MIDGE services:
REM   1. Ensures Docker/Qdrant is running
REM   2. Starts the main MIDGE orchestrator
REM
REM To run at Windows startup:
REM   1. Press Win+R, type: shell:startup
REM   2. Create shortcut to this .bat file in that folder
REM
REM Or use Task Scheduler:
REM   1. Open Task Scheduler
REM   2. Create Basic Task -> "MIDGE"
REM   3. Trigger: At startup
REM   4. Action: Start program -> this .bat file
REM ============================================================

echo ============================================================
echo MIDGE Startup Script
echo ============================================================
echo.

REM Change to MIDGE directory
cd /d C:\Users\baenb\projects\MIDGE

REM Check if Docker is running
echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [WARN] Docker not running. Attempting to start...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting 30 seconds for Docker to start...
    timeout /t 30 /nobreak >nul
)

REM Start Qdrant if not running
echo Checking Qdrant...
docker ps | findstr qdrant >nul 2>&1
if errorlevel 1 (
    echo Starting Qdrant container...
    docker start qdrant 2>nul || docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
    echo Waiting 10 seconds for Qdrant to initialize...
    timeout /t 10 /nobreak >nul
)

REM Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama not running. Please start Ollama manually.
    echo   Run: ollama serve
)

echo.
echo ============================================================
echo Starting MIDGE services...
echo ============================================================
echo.

REM Start MIDGE
python run.py

REM If MIDGE exits, pause to see any error messages
echo.
echo MIDGE has stopped.
pause
