@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set CFG=carla_root.txt
echo Fly-By-Wire — CARLA path setup
echo.

if exist "%CFG%" (
  echo Current %CFG%:
  type "%CFG%"
  echo.
)

set /p CARLA_IN=Enter full path to your CARLA_0.9.16 folder: 
if "%CARLA_IN%"=="" (
  echo Cancelled.
  pause
  exit /b 1
)

set "CARLA_IN=%CARLA_IN:"=%"
if not exist "%CARLA_IN%\CarlaUE4.exe" (
  echo.
  echo ERROR: CarlaUE4.exe not found at:
  echo   %CARLA_IN%
  echo.
  pause
  exit /b 1
)

> "%CFG%" echo %CARLA_IN%
echo.
echo Saved to demos\%CFG%
echo   %CARLA_IN%
echo.
echo Demos will use this path automatically. Optional: set CARLA_ROOT in your user environment.
pause
exit /b 0
