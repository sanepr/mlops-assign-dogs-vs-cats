#!/bin/bash
# Setup script for Cats vs Dogs MLOps Pipeline
# This script initializes the project environment

set -e

echo "=========================================="
echo "MLOps Pipeline Setup"
echo "Cats vs Dogs Classification"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Initialize Git if not already initialized
echo ""
echo "Checking Git initialization..."
if [ ! -d ".git" ]; then
    git init
    echo "Git initialized"
else
    echo "Git already initialized"
fi

# Initialize DVC (with error handling)
echo ""
echo "Checking DVC initialization..."
if [ ! -d ".dvc" ]; then
    if command -v dvc &> /dev/null; then
        dvc init || echo "DVC init failed - you can initialize manually later with 'dvc init'"
    else
        echo "DVC not found in PATH - skipping DVC initialization"
        echo "You can install it later with: pip install dvc"
    fi
else
    echo "DVC already initialized"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed models logs mlruns

# Create sample dataset
echo ""
echo "Creating sample dataset..."
python scripts/prepare_data.py --sample --source data/raw --output data/processed

# Run tests (optional, may fail without all dependencies)
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short 2>/dev/null || echo "Some tests may have failed - this is expected without full setup"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download the Cats vs Dogs dataset from Kaggle"
echo "   - Place images in data/raw/cats/ and data/raw/dogs/"
echo "   - Or run: python scripts/prepare_data.py --download"
echo ""
echo "2. Train the model:"
echo "   python src/train.py --data-dir data/processed/train --epochs 10"
echo ""
echo "3. Start the API server:"
echo "   python main.py"
echo ""
echo "4. Or use Docker Compose:"
echo "   docker-compose up -d"
echo ""
