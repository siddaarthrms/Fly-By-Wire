@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "recordings" mkdir "recordings"
if not exist "telemetry" mkdir "telemetry"

start "" "%CD%\recordings"
start "" "%CD%\telemetry"
exit /b 0
