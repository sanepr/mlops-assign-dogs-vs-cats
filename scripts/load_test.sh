#!/bin/bash
# Load Testing Script for Cats vs Dogs API
# This script generates traffic to populate Grafana dashboards

set -e

API_URL="${API_URL:-http://localhost:8000}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
PARALLEL="${PARALLEL:-5}"

echo "========================================"
echo "  Load Testing - Cats vs Dogs API"
echo "========================================"
echo "API URL: $API_URL"
echo "Number of requests: $NUM_REQUESTS"
echo "Parallel workers: $PARALLEL"
echo ""

# Check if API is healthy
echo "Checking API health..."
HEALTH=$(curl -s "$API_URL/health" 2>/dev/null || echo "failed")
if [[ "$HEALTH" == *"healthy"* ]]; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding. Please start the containers first."
    echo "   Run: docker-compose up -d"
    exit 1
fi

echo ""
echo "Starting load test..."
echo ""

# Function to make a single request
make_request() {
    local request_type=$1
    local request_num=$2

    case $request_type in
        "health")
            curl -s "$API_URL/health" > /dev/null
            ;;
        "stats")
            curl -s "$API_URL/stats" > /dev/null
            ;;
        "model_info")
            curl -s "$API_URL/model/info" > /dev/null
            ;;
        "metrics")
            curl -s "$API_URL/metrics" > /dev/null
            ;;
        "root")
            curl -s "$API_URL/" > /dev/null
            ;;
    esac
}

# Counter for requests
total_success=0
total_failed=0

# Run health checks
echo "📊 Running health checks..."
for i in $(seq 1 20); do
    if curl -s "$API_URL/health" > /dev/null 2>&1; then
        ((total_success++))
    else
        ((total_failed++))
    fi
done
echo "   Health checks: $total_success success, $total_failed failed"

# Run stats requests
echo "📊 Running stats requests..."
for i in $(seq 1 20); do
    if curl -s "$API_URL/stats" > /dev/null 2>&1; then
        ((total_success++))
    else
        ((total_failed++))
    fi
done
echo "   Stats requests completed"

# Run model info requests
echo "📊 Running model info requests..."
for i in $(seq 1 20); do
    if curl -s "$API_URL/model/info" > /dev/null 2>&1; then
        ((total_success++))
    else
        ((total_failed++))
    fi
done
echo "   Model info requests completed"

# Run root endpoint requests
echo "📊 Running root endpoint requests..."
for i in $(seq 1 20); do
    if curl -s "$API_URL/" > /dev/null 2>&1; then
        ((total_success++))
    else
        ((total_failed++))
    fi
done
echo "   Root endpoint requests completed"

# Run metrics requests
echo "📊 Running metrics requests..."
for i in $(seq 1 20); do
    if curl -s "$API_URL/metrics" > /dev/null 2>&1; then
        ((total_success++))
    else
        ((total_failed++))
    fi
done
echo "   Metrics requests completed"

# Create a test image and run predictions if possible
echo "📊 Running prediction requests..."
TEST_IMAGE="test_image.jpg"
if [ -f "$TEST_IMAGE" ]; then
    for i in $(seq 1 10); do
        if curl -s -X POST "$API_URL/predict" -F "file=@$TEST_IMAGE" > /dev/null 2>&1; then
            ((total_success++))
        else
            ((total_failed++))
        fi
    done
    echo "   Prediction requests completed"
else
    echo "   No test image found, skipping prediction tests"
fi

echo ""
echo "========================================"
echo "  Load Test Results"
echo "========================================"
echo "Total successful requests: $total_success"
echo "Total failed requests: $total_failed"
echo ""
echo "🎯 Check Grafana dashboard at: http://localhost:3000"
echo "   Default credentials: admin/admin"
echo ""
echo "📈 Check Prometheus at: http://localhost:9090"
echo "========================================"
