@echo off
:: MIDGE continuous service launcher
:: Compatible with NSSM or Windows Task Scheduler.
:: When run directly: restarts MIDGE automatically on exit (goto loop).

cd /d "%~dp0"

:loop
echo [%date% %time%] Starting MIDGE...
python main.py --continuous --agents 5 --steps 2000

echo [%date% %time%] MIDGE exited (code %ERRORLEVEL%). Restarting in 30 seconds...
timeout /t 30 /nobreak >nul
goto loop
