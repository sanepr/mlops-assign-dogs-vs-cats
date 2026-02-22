#!/bin/bash
# Smoke Test Script for Cats vs Dogs Classifier
# This script runs after deployment to verify the service is working

set -e

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8000}"
BASE_URL="http://${API_HOST}:${API_PORT}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "=========================================="
echo "Running Smoke Tests"
echo "Target: ${BASE_URL}"
echo "=========================================="

# Function to check if service is ready
wait_for_service() {
    echo "Waiting for service to be ready..."
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s -f "${BASE_URL}/health" > /dev/null 2>&1; then
            echo "✓ Service is ready!"
            return 0
        fi

        retries=$((retries + 1))
        echo "Attempt ${retries}/${MAX_RETRIES} - Service not ready, waiting ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done

    echo "✗ Service failed to become ready after ${MAX_RETRIES} attempts"
    return 1
}

# Test 1: Health Check
test_health_check() {
    echo ""
    echo "Test 1: Health Check"
    echo "--------------------"

    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        echo "✓ Health check passed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 0
    else
        echo "✗ Health check failed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 1
    fi
}

# Test 2: Root Endpoint
test_root_endpoint() {
    echo ""
    echo "Test 2: Root Endpoint"
    echo "---------------------"

    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        echo "✓ Root endpoint passed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 0
    else
        echo "✗ Root endpoint failed (HTTP ${http_code})"
        return 1
    fi
}

# Test 3: Stats Endpoint
test_stats_endpoint() {
    echo ""
    echo "Test 3: Stats Endpoint"
    echo "----------------------"

    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/stats")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        echo "✓ Stats endpoint passed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 0
    else
        echo "✗ Stats endpoint failed (HTTP ${http_code})"
        return 1
    fi
}

# Test 4: Model Info (may fail if model not loaded)
test_model_info() {
    echo ""
    echo "Test 4: Model Info Endpoint"
    echo "---------------------------"

    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/model/info")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        echo "✓ Model info passed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 0
    elif [ "$http_code" = "503" ]; then
        echo "⚠ Model not loaded (HTTP ${http_code}) - Expected if model file missing"
        return 0
    else
        echo "✗ Model info failed (HTTP ${http_code})"
        return 1
    fi
}

# Test 5: Metrics Endpoint (Prometheus)
test_metrics_endpoint() {
    echo ""
    echo "Test 5: Metrics Endpoint"
    echo "------------------------"

    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/metrics")
    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" = "200" ]; then
        echo "✓ Metrics endpoint passed (HTTP ${http_code})"
        return 0
    else
        echo "✗ Metrics endpoint failed (HTTP ${http_code})"
        return 1
    fi
}

# Test 6: Prediction Endpoint (with sample image)
test_prediction() {
    echo ""
    echo "Test 6: Prediction Endpoint"
    echo "---------------------------"

    # Create a simple test image (1x1 red pixel JPEG)
    # Base64 of a minimal valid JPEG
    IMAGE_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"image_base64\": \"${IMAGE_BASE64}\"}" \
        "${BASE_URL}/predict/base64")

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        echo "✓ Prediction endpoint passed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 0
    elif [ "$http_code" = "503" ]; then
        echo "⚠ Model not loaded (HTTP ${http_code}) - Prediction skipped"
        return 0
    else
        echo "✗ Prediction endpoint failed (HTTP ${http_code})"
        echo "  Response: ${body}"
        return 1
    fi
}

# Run all tests
main() {
    local failed=0

    wait_for_service || exit 1

    test_health_check || failed=$((failed + 1))
    test_root_endpoint || failed=$((failed + 1))
    test_stats_endpoint || failed=$((failed + 1))
    test_model_info || failed=$((failed + 1))
    test_metrics_endpoint || failed=$((failed + 1))
    test_prediction || failed=$((failed + 1))

    echo ""
    echo "=========================================="
    echo "Smoke Test Results"
    echo "=========================================="

    if [ $failed -eq 0 ]; then
        echo "✓ All smoke tests passed!"
        exit 0
    else
        echo "✗ ${failed} smoke test(s) failed"
        exit 1
    fi
}

main "$@"
