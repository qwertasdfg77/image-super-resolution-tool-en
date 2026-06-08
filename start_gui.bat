@echo off
setlocal
set "APP_DIR=%~dp0"
set "GUI_EXE=%APP_DIR%Image Super Resolution Tool.exe"
set "SCRIPT=%APP_DIR%dlss_like_super_resolution_gui.py"
set "VENV_PYTHONW=%APP_DIR%.venv\Scripts\pythonw.exe"
set "VENV_PYTHON=%APP_DIR%.venv\Scripts\python.exe"

if exist "%GUI_EXE%" (
    start "" "%GUI_EXE%"
    exit /b 0
)

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys, tkinter" >nul 2>nul
    if not errorlevel 1 (
        if exist "%VENV_PYTHONW%" (
            start "" "%VENV_PYTHONW%" "%SCRIPT%"
        ) else (
            start "" "%VENV_PYTHON%" "%SCRIPT%"
        )
        exit /b 0
    )
)

where pythonw >nul 2>nul
if not errorlevel 1 (
    pythonw -c "import sys, tkinter" >nul 2>nul
    if not errorlevel 1 (
        start "" pythonw "%SCRIPT%"
        exit /b 0
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys, tkinter" >nul 2>nul
    if not errorlevel 1 (
        start "" python "%SCRIPT%"
        exit /b 0
    )
)

echo The graphical launcher was not found. Please reinstall Image Super Resolution Tool.
pause
exit /b 1
