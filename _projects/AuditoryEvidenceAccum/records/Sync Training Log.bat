@echo off
REM Double-click this to pull the latest shared training_log.xlsx, refresh it from this box's own
REM local session data, then commit and push it back -- the full daily workflow in one step.
REM Only ever commits training_log.xlsx to main. Works on main OR on a computer-specific branch (e.g.
REM training05): on such a branch, main is merged in first, and only training_log.xlsx is pushed
REM to main -- the machine's own patches stay on its branch. See sync_training_log.py for details.
setlocal
cd /d "%~dp0"

set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if not exist "%PY_EXE%" set PY_EXE=C:\Users\Gil\.conda\envs\UM-pybpod\python.exe
if exist "%PY_EXE%" (
    "%PY_EXE%" sync_training_log.py
) else (
    call conda activate pybpod-environment
    python sync_training_log.py
)

:end
