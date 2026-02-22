#!/bin/bash
# Script to restart MLflow and verify experiments

cd /Users/aashishr/codebase/mlso_ass

echo "=== Stopping services ==="
docker-compose down

echo ""
echo "=== Checking mlruns directory ==="
ls -la mlruns/
ls -la mlruns/1/ 2>/dev/null || echo "No experiment 1"

echo ""
echo "=== Starting services ==="
docker-compose up -d

echo ""
echo "=== Waiting for services ==="
sleep 15

echo ""
echo "=== Checking container status ==="
docker-compose ps

echo ""
echo "=== Checking MLflow API ==="
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list 2>&1 | head -50

echo ""
echo "=== MLflow UI available at: http://localhost:5001 ==="
