@echo off
title BEACON v1.0.0 - IT Seminar
cd /d "%~dp0"
echo Open http://127.0.0.1:8000 in your browser.
echo Keep this window open. Press Ctrl+C to stop the server.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_release.py %*
) else (
    python scripts\run_release.py %*
)
if errorlevel 1 pause
