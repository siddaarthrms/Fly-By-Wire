@echo off
REM Helper for numbered demo launchers.
REM Launch: --opendrive-preset --use-neuron-throttle, ClearNoon, default vehicle.
REM Also: --chase-dynamic, --record-camera, MP4/telemetry resolution.
setlocal EnableExtensions
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8

set PRESET=%~1
set LABEL=%~2
set MAXSEC=%~3
set EXTRA=%~4
set CAM=%~5

if "%PRESET%"=="" (
  echo Usage: lib\run_demo.bat ^<preset^> ^<label^> [max_seconds] [extra pilot flags] [record_camera]
  exit /b 1
)
if "%LABEL%"=="" set LABEL=%PRESET%
if "%MAXSEC%"=="" set MAXSEC=45
if "%CAM%"=="" set CAM=chase_low

if not exist "demos\recordings" mkdir "demos\recordings"
if not exist "demos\telemetry" mkdir "demos\telemetry"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set OUT=demos\recordings\%LABEL%_%TS%.mp4

echo.
echo ============================================================
echo   Fly-By-Wire demo: %LABEL%
echo   Preset: %PRESET%  ^|  Auto-stop: %MAXSEC%s
echo   Chase camera: %CAM%  ^|  World: ClearNoon
echo   Recording: %OUT%
echo ============================================================
echo   CARLA will launch in a separate window.
echo   Press Ctrl+C to stop early.
echo.

python src\carla_pilot.py --launch --opendrive-preset %PRESET% --use-neuron-throttle --chase-dynamic --record-camera %CAM% --record-video "%OUT%" --record-max-seconds %MAXSEC% --cam-w 640 --cam-h 360 --record-w 1920 --record-h 1080 --print-every 30 %EXTRA%

set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo.
  echo Demo failed with exit code %ERR%.
  pause
  exit /b %ERR%
)

echo.
echo Demo complete.
echo Video: %OUT%
call "%~dp0..\Open.bat" outputs
pause
exit /b 0
