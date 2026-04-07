@echo off
:: ============================================================
::  AI Key Image Generator - Windows One-Click Setup & Run
:: ============================================================

echo.
echo ========================================
echo  AI Key Image Generator - Setup Start
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed.
    echo     Please install Python 3.10 or higher from:
    echo     https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% detected

:: Check Python version >= 3.10
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [X] Python %PYVER% is not supported.
    echo     Python 3.10 or higher is required.
    echo     Please install Python 3.10+ from:
    echo     https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

cd /d "%~dp0"

:: Create virtual environment (only if it does not exist)
if not exist "venv\" (
    echo.
    echo [..] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo.
echo [..] Installing required packages... ^(this may take a while on first run^)
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo [OK] Packages installed

:: Create outputs folder
if not exist "outputs\" mkdir outputs

echo.
echo ========================================
echo  Launching app!
echo  Open your browser at http://localhost:7860
echo  Press Ctrl+C or close this window to quit.
echo ========================================
echo.

python app.py

pause
