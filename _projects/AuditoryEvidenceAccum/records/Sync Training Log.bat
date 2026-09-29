@echo off
REM Double-click this to pull the latest shared training_log.xlsx, refresh it from this box's own
REM local session data, then commit and push it back -- the full daily workflow in one step.
REM Only ever commits training_log.xlsx to main. Works on main OR on a box-specific branch (e.g.
REM training-box-5): on a box branch, main is merged in first, and only training_log.xlsx is pushed
REM to main -- the box's own code stays on its branch. See sync_training_log.py for details.
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
