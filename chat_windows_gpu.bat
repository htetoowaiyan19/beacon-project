@echo off
title BEACON - Windows CUDA chat
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows_gpu.bat first.
    pause
    exit /b 1
)
echo Open http://127.0.0.1:8000/chat.html in your browser.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" scripts\run_laptop.py --chat %*
if errorlevel 1 pause
