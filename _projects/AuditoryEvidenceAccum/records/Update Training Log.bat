@echo off
REM Double-click this to refresh training_log.xlsx from this machine's own local session data.
REM Only touches this box's own subject sheet(s) -- safe to run any time, on any box.
setlocal
cd /d "%~dp0"

set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if exist "%PY_EXE%" (
    "%PY_EXE%" build_training_log.py
) else (
    call conda activate pybpod-environment
    python build_training_log.py
)

echo.
echo Done.
