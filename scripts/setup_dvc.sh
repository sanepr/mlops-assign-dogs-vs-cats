#!/bin/bash
# Script to initialize DVC separately
# Run this after the main setup.sh if you need DVC functionality

echo "=== DVC Setup ==="

# Check if in virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Install DVC with specific compatible versions
echo ""
echo "Installing DVC..."
pip install "dvc>=3.55.0" "pathspec>=0.12.1" --upgrade

# Verify installation
echo ""
echo "Verifying DVC installation..."
dvc version

# Initialize DVC if not already initialized
echo ""
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    echo "DVC initialized successfully"
else
    echo "DVC already initialized"
fi

# Add data to DVC tracking
echo ""
echo "To track your data with DVC, run:"
echo "  dvc add data/raw"
echo "  git add data/raw.dvc .gitignore"
echo "  git commit -m 'Add raw data to DVC'"

echo ""
echo "=== DVC Setup Complete ==="
