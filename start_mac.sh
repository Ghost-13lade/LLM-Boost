#!/bin/bash
# LLM Boost Launcher for macOS/Linux
# Double-click this file or run: ./start_mac.sh

echo "🚀 Launching LLM Boost..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 First-time setup: Creating virtual environment..."
    python3 -m venv venv
    
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    echo ""
    echo "🧠 Running setup wizard..."
    python3 setup_wizard.py
else
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "⚡ Starting Streamlit server..."
echo ""

# Launch Streamlit
streamlit run app.py

# If Streamlit exits, deactivate the virtual environment
deactivate 2>/dev/null