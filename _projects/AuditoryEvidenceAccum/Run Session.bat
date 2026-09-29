@echo off
REM Double-click this to open the session runner -- launch a stage task, watch its live progress,
REM record weight/notes -- without needing the PyBpod GUI open at all.
REM
REM Runs with a normal (not windowless) python.exe on purpose: pythonw.exe silently swallows any
REM startup error (missing dependency, bad path, ...) with no visible sign at all, which is exactly
REM why double-clicking could appear to "do nothing" -- this keeps a console window open so any
REM failure is actually visible instead of silent.
setlocal
cd /d "%~dp0"

REM Tries each known pybpod-capable environment in order (same "edit per box" convention as
REM VAR_TASK_PYTHON_EXE inside run_session.py itself) -- the real rig's own conda env is named
REM "pybpod-environment"; this dev machine's is the Anaconda env "UM-pybpod", confirmed to already
REM have every dependency this GUI needs (PyQt5/cv2/matplotlib/openpyxl) alongside pybpodapi
REM itself, since it's the same env PyBpod's own GUI runs from.
set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if exist "%PY_EXE%" goto :run

set PY_EXE=C:\Users\Gil\anaconda3\envs\UM-pybpod\python.exe
if exist "%PY_EXE%" goto :run

where python >nul 2>nul
if %errorlevel%==0 (
    set PY_EXE=python
    goto :run
)

echo.
echo ERROR: could not find a Python interpreter.
echo Looked for the known pybpod environments and "python" on PATH.
echo Edit this file and set PY_EXE to your pybpod environment's actual python.exe location.
echo.
pause
goto :eof

:run
"%PY_EXE%" run_session.py
if errorlevel 1 (
    echo.
    echo run_session.py exited with an error -- see the message above.
    echo.
    pause
)
