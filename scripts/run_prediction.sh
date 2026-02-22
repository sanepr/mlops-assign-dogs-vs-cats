#!/bin/bash
# Test prediction script

cd /Users/aashishr/codebase/mlso_ass

echo "Starting containers..."
docker-compose up -d

echo "Waiting for API to be ready..."
sleep 20

echo "Checking health..."
curl -s http://localhost:8000/health > /Users/aashishr/codebase/mlso_ass/logs/health_output.txt

echo "Making prediction..."
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  > /Users/aashishr/codebase/mlso_ass/logs/prediction_output.txt

echo "Results saved to logs/"
cat /Users/aashishr/codebase/mlso_ass/logs/prediction_output.txt
