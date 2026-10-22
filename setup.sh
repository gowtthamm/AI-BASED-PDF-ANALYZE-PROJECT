#!/usr/bin/env bash
# setup.sh
# One-step setup: creates a virtual environment, installs dependencies,
# and prepares the .env file. Run this once after opening the project.
#
# Usage:
#   chmod +x setup.sh && ./setup.sh

set -e

# Detect Python command (on Windows, 'python' or 'py' is used; 'python3' is often a Microsoft Store shim)
if command -v python >/dev/null 2>&1; then
    PY_CMD="python"
elif command -v py >/dev/null 2>&1; then
    PY_CMD="py"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
else
    echo "Error: Python executable not found. Please install Python or ensure it is added to your PATH."
    exit 1
fi

echo "Creating virtual environment (venv) using $PY_CMD..."
$PY_CMD -m venv venv

echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "Creating .env template..."
        echo "GROQ_API_KEY=" > .env
    fi
    echo "IMPORTANT: Open .env and add your GROQ_API_KEY before running the app."
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. Add your Groq API key to the .env file."
if [ -f "venv/Scripts/activate" ]; then
    echo "  2. Run: source venv/Scripts/activate && streamlit run app.py"
    echo "     (or in PowerShell: .\\venv\\Scripts\\Activate.ps1; streamlit run app.py)"
else
    echo "  2. Run: source venv/bin/activate && streamlit run app.py"
fi
echo "     (or just press F5 in VS Code)"
