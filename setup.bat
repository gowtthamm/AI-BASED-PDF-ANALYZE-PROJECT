@echo off
echo ============================================
echo AI-Based Preparation Guide - Setup Script
echo ============================================

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ from https://python.org and make sure "Add python.exe to PATH" is checked.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment (venv)...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Setting up environment variables...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo Created .env from .env.example
    ) else (
        echo GROQ_API_KEY= > .env
        echo Created empty .env file
    )
    echo IMPORTANT: Open .env and add your GROQ_API_KEY before running the app!
) else (
    echo .env file already exists.
)

echo.
echo ============================================
echo Setup complete!
echo Next steps:
echo   1. Open .env and insert your GROQ_API_KEY.
echo   2. Run: venv\Scripts\activate ^&^& streamlit run app.py
echo      (or press F5 in VS Code)
echo ============================================
echo.
pause
