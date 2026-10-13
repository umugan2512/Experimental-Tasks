@echo off
REM Double-click this to pull the latest main, refresh training_log.xlsx from this box's own local
REM session data, then commit the training data (training_log.xlsx + new session folders) to main and
REM push it -- the full daily workflow in one step. Works on main OR on a computer-specific branch
REM (e.g. training05): only data ever goes to main from such a branch; the machine's own patches
REM stay on it. See sync_training_log.py for details.
setlocal
cd /d "%~dp0"

set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if exist "%PY_EXE%" (
    "%PY_EXE%" sync_training_log.py
) else (
    call conda activate pybpod-environment
    python sync_training_log.py
)

:end
