@echo off
title BEACON Burmese AI Server
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_server.py %*
) else (
    python scripts\run_server.py %*
)
if errorlevel 1 pause
