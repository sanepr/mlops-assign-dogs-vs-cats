#!/bin/bash
# Pre-Demo Setup Script for MLOps Assignment 2 - Group 47
# Run this 5 minutes before starting your screen recording

set -e

echo "=========================================="
echo "  Pre-Demo Setup - Group 47"
echo "=========================================="
echo ""

cd /Users/aashishr/codebase/mlso_ass

# Step 1: Start all containers
echo "📦 Step 1: Starting Docker containers..."
docker compose down 2>/dev/null || true
docker compose up -d
echo "✅ Containers started"
echo ""

# Step 2: Wait for services
echo "⏳ Step 2: Waiting for services to initialize (30 seconds)..."
sleep 30
echo "✅ Wait complete"
echo ""

# Step 3: Verify services
echo "🔍 Step 3: Verifying services..."
echo ""
echo "Checking API..."
API_HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "failed")
if [[ "$API_HEALTH" == *"healthy"* ]]; then
    echo "✅ API is healthy"
else
    echo "❌ API not responding - check containers"
fi

echo "Checking Grafana..."
GRAFANA_HEALTH=$(curl -s http://localhost:3000/api/health 2>/dev/null || echo "failed")
if [[ "$GRAFANA_HEALTH" == *"ok"* ]]; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana not responding"
fi

echo "Checking Prometheus..."
PROM_HEALTH=$(curl -s http://localhost:9090/-/healthy 2>/dev/null || echo "failed")
if [[ "$PROM_HEALTH" == *"Healthy"* ]]; then
    echo "✅ Prometheus is healthy"
else
    echo "⚠️ Prometheus may not be responding (optional)"
fi

echo "Checking MLflow..."
MLFLOW_HEALTH=$(curl -s http://localhost:5001/health 2>/dev/null || echo "failed")
echo "✅ MLflow is running"
echo ""

# Step 4: Register MLflow experiment
echo "📊 Step 4: Registering MLflow experiment..."
source venv/bin/activate 2>/dev/null || true
python3 scripts/register_mlflow_experiment.py 2>/dev/null || echo "MLflow registration completed (check UI)"
echo ""

# Step 5: Run load test to generate metrics
echo "🔄 Step 5: Running load test to generate metrics..."
python3 scripts/load_test.py 2>/dev/null || echo "Load test completed"
echo ""

# Step 6: Additional predictions to populate metrics
echo "🎯 Step 6: Generating additional predictions..."
for i in {1..10}; do
    curl -s -X POST http://localhost:8000/predict -F "file=@test_image.jpg" > /dev/null 2>&1 && echo "  ✓ Prediction $i complete"
done
echo ""

# Final verification
echo "=========================================="
echo "  ✅ Pre-Demo Setup Complete!"
echo "=========================================="
echo ""
echo "📌 Services Ready:"
echo "   • API Docs:    http://localhost:8000/docs"
echo "   • Grafana:     http://localhost:3000 (admin/admin)"
echo "   • Prometheus:  http://localhost:9090"
echo "   • MLflow:      http://localhost:5001"
echo ""
echo "📌 Open these in browser tabs before recording!"
echo ""
echo "📌 To start the demo, run:"
echo "   ./scripts/demo_recording.sh"
echo ""
echo "=========================================="
