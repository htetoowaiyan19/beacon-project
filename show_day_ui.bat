@echo off
title BEACON - Show day control
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\show_day_ui.py
) else (
    python scripts\show_day_ui.py
)
if errorlevel 1 pause
