@echo off
REM Double-click this to reset session_startup_checklist.xlsx / .md back to a blank template.
setlocal
cd /d "%~dp0"

set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if exist "%PY_EXE%" (
    "%PY_EXE%" build_startup_checklist.py
) else (
    call conda activate pybpod-environment
    python build_startup_checklist.py
)

echo.
echo Done -- press any key to close this window.
pause >nul
