@echo off
REM Double-click this to pull the latest shared training_log.xlsx, refresh it from this box's own
REM local session data, then commit and push it back -- the full daily workflow in one step.
REM Only ever touches training_log.xlsx -- never any other file, never any code.
setlocal
cd /d "%~dp0"

echo Pulling latest changes...
git pull
if errorlevel 1 (
    echo.
    echo ERROR: git pull failed -- see the message above ^(often a conflict that needs resolving by
    echo hand, since training_log.xlsx is a binary file git can't auto-merge^). Nothing was updated
    echo or pushed. Fix this first, then run again.
    goto :end
)

set PY_EXE=C:\Users\2P-Behav\.conda\envs\pybpod-environment\python.exe
if exist "%PY_EXE%" (
    "%PY_EXE%" build_training_log.py
) else (
    call conda activate pybpod-environment
    python build_training_log.py
)
if errorlevel 1 (
    echo.
    echo ERROR: build_training_log.py failed -- see the message above. Nothing was committed.
    goto :end
)

git diff --quiet -- training_log.xlsx
if errorlevel 1 (
    echo.
    echo Committing and pushing training_log.xlsx...
    git add training_log.xlsx
    git commit -m "Update training_log.xlsx from %COMPUTERNAME%"
    git push
    if errorlevel 1 (
        echo.
        echo ERROR: git push failed -- your commit is saved locally but not shared yet. Someone else
        echo may have pushed in the meantime -- try running this again.
        goto :end
    )
    echo.
    echo Done -- training_log.xlsx updated, committed, and pushed.
) else (
    echo.
    echo No changes to training_log.xlsx since the last sync -- nothing to commit.
)

:end
