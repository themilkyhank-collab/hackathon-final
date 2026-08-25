@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Prefer the Windows Python launcher, then fall back to python.exe.
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3 was not found. Install Python 3.11+ and run start.bat again.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

rem Do not create a virtual environment: use the user's existing Python installation.
rem Install dependencies only when the required imports are missing.
echo Checking Python dependencies...
%PYTHON% -c "import fastapi,uvicorn,pandas,numpy,sklearn,scipy,joblib" >nul 2>nul
if errorlevel 1 (
    echo Installing missing dependencies...
    %PYTHON% -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
    if errorlevel 1 goto :error
)

echo Starting Aesteel on localhost...
%PYTHON% start.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Aesteel could not be started. See the error above.
pause
exit /b 1
