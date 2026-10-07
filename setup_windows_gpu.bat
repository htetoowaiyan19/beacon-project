@echo off
title BEACON - Windows GGUF setup
cd /d "%~dp0"
python scripts\setup_windows_gpu.py %*
if errorlevel 1 pause
