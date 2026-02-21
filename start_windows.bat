@echo off
REM LLM Boost Launcher for Windows
REM Double-click this file to run

echo.
echo 🚀 Launching LLM Boost...
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist venv (
    echo 📦 First-time setup: Creating virtual environment...
    python -m venv venv
    
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate
    
    echo ⬆️  Upgrading pip...
    pip install --upgrade pip
    
    echo.
    echo 🧠 Running setup wizard...
    python setup_wizard.py
) else (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate
)

echo.
echo ⚡ Starting Streamlit server...
echo.

REM Launch Streamlit
streamlit run app.py

REM If Streamlit exits, deactivate the virtual environment
deactivate 2>nul

echo.
echo Press any key to exit...
pause >nul