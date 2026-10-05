@echo off
title BEACON Training
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\training_ui.py
) else (
    python scripts\training_ui.py
)
if errorlevel 1 pause
