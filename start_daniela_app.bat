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

set "PYTHON_CMD="

REM Prefer the py launcher — avoids the Windows Store python.exe alias trap.
where py >nul 2>nul
if not errorlevel 1 (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

REM Fall back to python only if it is a real install, not the Store stub.
if "%PYTHON_CMD%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        python --version >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)

REM Common per-user install path (python.org installer on Windows).
if "%PYTHON_CMD%"=="" (
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
        set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
    ) else if exist "%LocalAppData%\Python\pythoncore-3.14-64\python.exe" (
        set "PYTHON_CMD=%LocalAppData%\Python\pythoncore-3.14-64\python.exe"
    ) else if exist "%LocalAppData%\Programs\Python\Python314\python.exe" (
        set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python314\python.exe"
    )
)

if "%PYTHON_CMD%"=="" (
    echo Python was not found.
    echo.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo Check "Add python.exe to PATH" during install, then run this script again.
    echo.
    echo If Python is already installed, disable the Store alias:
    echo Settings ^> Apps ^> Advanced app settings ^> App execution aliases
    echo Turn OFF "python.exe" and "python3.exe", then reopen this window.
    pause
    exit /b 1
)

echo Using: %PYTHON_CMD%
%PYTHON_CMD% --version

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
