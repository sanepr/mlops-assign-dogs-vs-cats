#!/usr/bin/env python3
"""Load Testing Script for Cats vs Dogs API"""

import requests
import time
import sys
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")
NUM_REQUESTS = 100

print("=" * 50)
print("  Load Testing - Cats vs Dogs API")
print("=" * 50)
print(f"API URL: {API_URL}")
print(f"Number of requests per endpoint: {NUM_REQUESTS // 5}")
print()

# Check if API is healthy
print("Checking API health...")
try:
    resp = requests.get(f"{API_URL}/health", timeout=5)
    if resp.status_code == 200:
        print("✅ API is healthy")
        print(f"   Response: {resp.json()}")
    else:
        print(f"❌ API returned status {resp.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"❌ API is not responding: {e}")
    print("   Please start the containers first:")
    print("   docker-compose up -d")
    sys.exit(1)

print()
print("Starting load test...")
print()

total_success = 0
total_failed = 0
total_time = 0

# Test endpoints
endpoints = [
    ("/health", "GET", None, "Health Check"),
    ("/stats", "GET", None, "Stats"),
    ("/model/info", "GET", None, "Model Info"),
    ("/", "GET", None, "Root"),
    ("/metrics", "GET", None, "Metrics"),
]

for path, method, data, name in endpoints:
    print(f"📊 Testing {name} ({path})...")
    success = 0
    failed = 0
    times = []

    for i in range(NUM_REQUESTS // 5):
        try:
            start = time.time()
            if method == "GET":
                resp = requests.get(f"{API_URL}{path}", timeout=5)
            else:
                resp = requests.post(f"{API_URL}{path}", json=data, timeout=5)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

            if resp.status_code == 200:
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

    avg_time = sum(times) / len(times) if times else 0
    print(f"   ✅ Success: {success}, ❌ Failed: {failed}, ⏱️ Avg: {avg_time:.2f}ms")
    total_success += success
    total_failed += failed
    total_time += sum(times)

# Test prediction endpoint with test image if available
test_image_path = "test_image.jpg"
if os.path.exists(test_image_path):
    print(f"📊 Testing Prediction ({test_image_path})...")
    success = 0
    failed = 0
    times = []

    for i in range(10):
        try:
            start = time.time()
            with open(test_image_path, "rb") as f:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    timeout=30
                )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

            if resp.status_code == 200:
                success += 1
                if i == 0:
                    print(f"   Sample response: {resp.json()}")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"   Error: {e}")

    avg_time = sum(times) / len(times) if times else 0
    print(f"   ✅ Success: {success}, ❌ Failed: {failed}, ⏱️ Avg: {avg_time:.2f}ms")
    total_success += success
    total_failed += failed
else:
    print(f"⚠️ No test image found at {test_image_path}")

print()
print("=" * 50)
print("  Load Test Results")
print("=" * 50)
print(f"Total successful requests: {total_success}")
print(f"Total failed requests: {total_failed}")
print(f"Success rate: {(total_success / (total_success + total_failed)) * 100:.1f}%")
print()
print("🎯 Check Grafana dashboard at: http://localhost:3000")
print("   Default credentials: admin/admin")
print()
print("📈 Check Prometheus at: http://localhost:9090")
print("=" * 50)
