#!/bin/bash
# Demo Script for MLOps Assignment 2 - Group 47
# Run this during screen recording for smooth demonstration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}>>> $1${NC}"
}

pause() {
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# Start Demo
clear
print_header "🎬 MLOps Demo - Group 47"
echo "Cats vs Dogs Classification Pipeline"
echo "This demo showcases the complete MLOps workflow"
pause

# M1: Model Development
print_header "M1: Model Development & Experiment Tracking"

print_step "Showing DVC Pipeline Configuration"
cat dvc.yaml
pause

print_step "Showing Model Architecture (first 30 lines)"
head -30 src/model.py
pause

print_step "MLflow is running at http://localhost:5001"
echo "Open in browser to see experiments"
pause

# M2: Containerization
print_header "M2: Model Packaging & Containerization"

print_step "Showing Dockerfile"
head -40 Dockerfile
pause

print_step "Showing Running Containers"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
pause

print_step "API Documentation available at http://localhost:8000/docs"
pause

# M3: CI Pipeline
print_header "M3: CI Pipeline"

print_step "Showing CI Pipeline Structure"
grep -E "^  [a-z]+:|name:" .github/workflows/ci.yml | head -30
pause

print_step "CI Pipeline includes:"
echo "  1. Code Linting (flake8, black, isort)"
echo "  2. Unit Tests (pytest)"
echo "  3. Train ML Model"
echo "  4. Build Docker Image"
echo "  5. Push to Registry"
echo "  6. Security Scan"
pause

# M4: CD Pipeline
print_header "M4: CD Pipeline & Deployment"

print_step "Showing Kubernetes Deployment"
head -35 k8s/deployment.yaml
pause

print_step "Showing Smoke Test Script"
head -40 scripts/smoke_test.sh
pause

# M5: Live Demo
print_header "M5: Live Prediction Demo"

print_step "1. Health Check"
curl -s http://localhost:8000/health | python3 -m json.tool
pause

print_step "2. Making a Prediction"
echo "Sending test_image.jpg to /predict endpoint..."
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" | python3 -m json.tool
pause

print_step "3. Service Statistics"
curl -s http://localhost:8000/stats | python3 -m json.tool
pause

print_step "4. Running Quick Load Test (5 requests)"
for i in {1..5}; do
    curl -s -X POST http://localhost:8000/predict -F "file=@test_image.jpg" > /dev/null
    echo "  ✓ Request $i complete"
done
pause

print_step "5. Checking ML Metrics"
echo ""
echo "Fetching metrics from /metrics endpoint..."
echo ""
# Get specific metrics
echo "📊 HTTP Request Metrics:"
curl -s http://localhost:8000/metrics | grep "http_requests_total" | head -5
echo ""
echo "📊 Model Prediction Metrics:"
curl -s http://localhost:8000/metrics | grep "model_predictions_total" | head -3
curl -s http://localhost:8000/metrics | grep "model_accuracy" | head -1
curl -s http://localhost:8000/metrics | grep "model_average_confidence" | head -1
curl -s http://localhost:8000/metrics | grep "model_cat_predictions" | head -1
curl -s http://localhost:8000/metrics | grep "model_dog_predictions" | head -1
echo ""
echo "📊 Inference Time:"
curl -s http://localhost:8000/metrics | grep "model_inference_seconds" | head -2
pause

# Monitoring
print_header "M5: Monitoring Dashboards"

echo "📊 Grafana Dashboard: http://localhost:3000"
echo "   Credentials: admin/admin"
echo ""
echo "📈 Prometheus: http://localhost:9090"
echo ""
echo "🔬 MLflow: http://localhost:5001"
pause

# Conclusion
print_header "🎉 Demo Complete!"
echo ""
echo "We demonstrated:"
echo "  ✅ M1: Model Development with MLflow & DVC"
echo "  ✅ M2: FastAPI Service & Docker Containerization"
echo "  ✅ M3: CI Pipeline with GitHub Actions"
echo "  ✅ M4: CD Pipeline & Kubernetes Deployment"
echo "  ✅ M5: Prometheus + Grafana Monitoring"
echo ""
echo "Thank you for watching - Group 47"
echo ""
