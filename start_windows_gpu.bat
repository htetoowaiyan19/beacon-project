@echo off
title BEACON - Windows CUDA members panel
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows_gpu.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\run_laptop.py %*
if errorlevel 1 pause
