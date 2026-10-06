@echo off
title BEACON - Device setup
cd /d "%~dp0"
python scripts\setup_device.py %*
if errorlevel 1 pause
