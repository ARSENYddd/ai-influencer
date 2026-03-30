#!/bin/bash
# Setup script for AI Influencer project

set -e

echo "=== AI Influencer Setup ==="

# Check Python version
PYTHON=$(which python3.11 2>/dev/null || which python3 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11+ not found. Install it first."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Using Python: $PY_VERSION at $PYTHON"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env from template if not exists
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "Created .env from template. Fill in your credentials!"
else
    echo ".env already exists — skipping."
fi

# Create output directories
mkdir -p output/images output/logs
touch output/images/.gitkeep output/logs/.gitkeep

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API credentials"
echo "  2. Test with:  python pipeline.py --dry-run"
echo "  3. Run live:   python pipeline.py"
echo "  4. Schedule:   python scheduler.py"
echo ""
