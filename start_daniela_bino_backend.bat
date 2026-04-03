@echo off
setlocal
cd /d "%~dp0"

if "%BINO_HOST%"=="" set BINO_HOST=127.0.0.1
if "%BINO_PORT%"=="" set BINO_PORT=8008

python -m uvicorn daniela_bino_backend:app --host %BINO_HOST% --port %BINO_PORT%
