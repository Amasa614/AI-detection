@echo off
setlocal
cd /d "%~dp0"

set "BINO_HOST=%BINO_HOST%"
if "%BINO_HOST%"=="" set "BINO_HOST=127.0.0.1"

set "BINO_PORT=%BINO_PORT%"
if "%BINO_PORT%"=="" set "BINO_PORT=8008"

set "FRONTEND_HOST=%FRONTEND_HOST%"
if "%FRONTEND_HOST%"=="" set "FRONTEND_HOST=127.0.0.1"

set "FRONTEND_PORT=%FRONTEND_PORT%"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=8010"

set "PYTHON_CMD=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found in PATH.
        echo Install Python, then run this script again.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=py -3"
)

>render-config.js echo window.BINO_BACKEND_URL = 'http://%BINO_HOST%:%BINO_PORT%';

echo Starting backend on http://%BINO_HOST%:%BINO_PORT% ...
start "Daniela Backend" cmd /k %PYTHON_CMD% -m uvicorn daniela_bino_backend:app --host %BINO_HOST% --port %BINO_PORT%

echo Starting frontend on http://%FRONTEND_HOST%:%FRONTEND_PORT%/daniela_ai_detection.html ...
start "Daniela Frontend" cmd /k %PYTHON_CMD% -m http.server %FRONTEND_PORT% --bind %FRONTEND_HOST%

timeout /t 2 /nobreak >nul

start "" "http://%FRONTEND_HOST%:%FRONTEND_PORT%/daniela_ai_detection.html"

echo.
echo Backend:  http://%BINO_HOST%:%BINO_PORT%
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%/daniela_ai_detection.html
echo.
echo Close the two command windows when you want to stop the app.
