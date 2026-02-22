#!/bin/bash
# Test script to verify all README instructions work
# Run this from the project root directory

set -e

LOG_FILE="test_readme_results.log"
echo "=== README Instructions Test ===" > $LOG_FILE
echo "Date: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

# Function to log results
log_test() {
    echo "[$1] $2" | tee -a $LOG_FILE
}

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    log_test "PASS" "Virtual environment activated"
else
    log_test "INFO" "No venv found, using system Python"
fi

echo "" | tee -a $LOG_FILE
echo "=== Testing Quick Start Instructions ===" | tee -a $LOG_FILE

# Test 1: Python version
echo "" | tee -a $LOG_FILE
echo "Test 1: Python version" | tee -a $LOG_FILE
if python3 --version >> $LOG_FILE 2>&1; then
    log_test "PASS" "Python version check"
else
    log_test "FAIL" "Python version check"
fi

# Test 2: Install dependencies
echo "" | tee -a $LOG_FILE
echo "Test 2: Install dependencies" | tee -a $LOG_FILE
if pip install -r requirements.txt -q 2>> $LOG_FILE; then
    log_test "PASS" "Dependencies installed"
else
    log_test "FAIL" "Dependencies installation failed"
fi

# Test 3: Prepare sample data
echo "" | tee -a $LOG_FILE
echo "Test 3: Prepare sample data" | tee -a $LOG_FILE
if python scripts/prepare_data.py --sample --source data/raw --output data/processed >> $LOG_FILE 2>&1; then
    log_test "PASS" "Sample data prepared"
else
    log_test "FAIL" "Sample data preparation failed"
fi

# Test 4: Check data exists
echo "" | tee -a $LOG_FILE
echo "Test 4: Check data exists" | tee -a $LOG_FILE
if [ -d "data/processed/train" ]; then
    TRAIN_COUNT=$(find data/processed/train -name "*.jpg" | wc -l)
    log_test "PASS" "Training data exists ($TRAIN_COUNT images)"
else
    log_test "FAIL" "Training data not found"
fi

# Test 5: DVC init (already initialized)
echo "" | tee -a $LOG_FILE
echo "Test 5: DVC initialization" | tee -a $LOG_FILE
if [ -d ".dvc" ]; then
    log_test "PASS" "DVC already initialized"
else
    if dvc init >> $LOG_FILE 2>&1; then
        log_test "PASS" "DVC initialized"
    else
        log_test "WARN" "DVC init failed (optional)"
    fi
fi

# Test 6: Run pytest
echo "" | tee -a $LOG_FILE
echo "Test 6: Run unit tests" | tee -a $LOG_FILE
if pytest tests/ -v --tb=short >> $LOG_FILE 2>&1; then
    log_test "PASS" "All unit tests passed"
else
    log_test "WARN" "Some tests failed (check log)"
fi

# Test 7: Build Docker image
echo "" | tee -a $LOG_FILE
echo "Test 7: Docker build" | tee -a $LOG_FILE
if command -v docker &> /dev/null; then
    if docker build -t cats-dogs-classifier:test . >> $LOG_FILE 2>&1; then
        log_test "PASS" "Docker image built"
    else
        log_test "FAIL" "Docker build failed"
    fi
else
    log_test "SKIP" "Docker not installed"
fi

# Test 8: MLflow import
echo "" | tee -a $LOG_FILE
echo "Test 8: MLflow import" | tee -a $LOG_FILE
if python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')" >> $LOG_FILE 2>&1; then
    log_test "PASS" "MLflow imported successfully"
else
    log_test "FAIL" "MLflow import failed"
fi

# Test 9: TensorFlow import
echo "" | tee -a $LOG_FILE
echo "Test 9: TensorFlow import" | tee -a $LOG_FILE
if python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" >> $LOG_FILE 2>&1; then
    log_test "PASS" "TensorFlow imported successfully"
else
    log_test "FAIL" "TensorFlow import failed"
fi

# Test 10: FastAPI import
echo "" | tee -a $LOG_FILE
echo "Test 10: FastAPI import" | tee -a $LOG_FILE
if python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" >> $LOG_FILE 2>&1; then
    log_test "PASS" "FastAPI imported successfully"
else
    log_test "FAIL" "FastAPI import failed"
fi

# Test 11: Model creation
echo "" | tee -a $LOG_FILE
echo "Test 11: Model creation" | tee -a $LOG_FILE
if python -c "from src.model import create_simple_cnn; m = create_simple_cnn(); print(f'Model created with {m.count_params()} params')" >> $LOG_FILE 2>&1; then
    log_test "PASS" "Model created successfully"
else
    log_test "FAIL" "Model creation failed"
fi

# Test 12: API schemas
echo "" | tee -a $LOG_FILE
echo "Test 12: API schemas" | tee -a $LOG_FILE
if python -c "from api.schemas import HealthResponse, PredictionResponse; print('Schemas OK')" >> $LOG_FILE 2>&1; then
    log_test "PASS" "API schemas imported successfully"
else
    log_test "FAIL" "API schemas import failed"
fi

# Summary
echo "" | tee -a $LOG_FILE
echo "=== TEST SUMMARY ===" | tee -a $LOG_FILE
PASS_COUNT=$(grep -c "\[PASS\]" $LOG_FILE || echo 0)
FAIL_COUNT=$(grep -c "\[FAIL\]" $LOG_FILE || echo 0)
WARN_COUNT=$(grep -c "\[WARN\]" $LOG_FILE || echo 0)
SKIP_COUNT=$(grep -c "\[SKIP\]" $LOG_FILE || echo 0)

echo "PASSED: $PASS_COUNT" | tee -a $LOG_FILE
echo "FAILED: $FAIL_COUNT" | tee -a $LOG_FILE
echo "WARNINGS: $WARN_COUNT" | tee -a $LOG_FILE
echo "SKIPPED: $SKIP_COUNT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Full log saved to: $LOG_FILE" | tee -a $LOG_FILE
