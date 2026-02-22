# MLOps Demo Recording Script
## Group 47 - Cats vs Dogs Classification
### Duration: ~5 minutes

---

## 🎬 Pre-Recording Setup (Run These Commands First!)

**Run these commands 5 minutes before recording to ensure everything is working:**

```bash
# 1. Start all containers
cd /Users/aashishr/codebase/mlso_ass
docker compose up -d

# 2. Wait for services to start
sleep 20

# 3. Verify all services are running
docker ps

# 4. Register fresh MLflow experiment (creates new run with metrics)
source venv/bin/activate
python3 scripts/register_mlflow_experiment.py

# 5. Generate API metrics by running load test
python3 scripts/load_test.py

# 6. Verify services are accessible
curl -s http://localhost:8000/health | jq
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list
```

---

## 🎬 Pre-Recording Checklist

Before starting the recording, ensure:
- [ ] All containers are running: `docker compose up -d`
- [ ] MLflow experiment registered: `python3 scripts/register_mlflow_experiment.py`
- [ ] Load test completed: `python3 scripts/load_test.py`
- [ ] Terminal is clean and font size is readable
- [ ] Browser tabs ready for: Grafana, Prometheus, MLflow, API Docs
- [ ] VS Code/IDE open with the project
- [ ] Test image ready (`test_image.jpg`)

---

## 📹 Recording Script

### **[0:00 - 0:30] Introduction & Project Overview**

**Show:** README.md in IDE

**Say:**
> "Hello, this is Group 47 presenting our MLOps pipeline for Cats vs Dogs image classification. This project demonstrates a complete end-to-end MLOps workflow including model development, containerization, CI/CD, and monitoring."

**Action:** Scroll through README showing project structure

---

### **[0:30 - 1:00] M1: Model Development & Experiment Tracking**

**Show:** Terminal

**Commands to run:**
```bash
# Show project structure
ls -la

# Show DVC configuration
cat dvc.yaml

# Show model architecture
head -50 src/model.py
```

**Show:** MLflow UI (http://localhost:5001)

**Say:**
> "We use DVC for data versioning and MLflow for experiment tracking. Here you can see our experiments with logged parameters, metrics, and model artifacts."

**Action:** Click on an experiment run, show parameters and metrics

---

### **[1:00 - 1:45] M2: Model Packaging & Containerization**

**Show:** Terminal

**Commands to run:**
```bash
# Show Dockerfile
cat Dockerfile

# Show docker-compose
cat docker-compose.yml

# Show running containers
docker ps
```

**Show:** API Swagger UI (http://localhost:8000/docs)

**Say:**
> "The model is packaged as a FastAPI REST service and containerized using Docker. We have multiple endpoints including health check, prediction, and metrics."

**Action:** Show the API endpoints in Swagger UI

---

### **[1:45 - 2:30] M3: CI Pipeline Demo**

**Show:** GitHub Actions in browser OR `.github/workflows/ci.yml` in IDE

**Commands to run:**
```bash
# Show CI pipeline
cat .github/workflows/ci.yml | head -80
```

**Say:**
> "Our CI pipeline includes 6 steps: Code Linting, Unit Tests, Model Training, Docker Build, Push to Registry, and Security Scan. Let me show the pipeline execution."

**Show:** GitHub Actions workflow run (screenshot or live)

**Action:** Show pipeline stages completing successfully

---

### **[2:30 - 3:15] M4: CD Pipeline & Deployment**

**Show:** Terminal

**Commands to run:**
```bash
# Show CD pipeline
cat .github/workflows/cd.yml | head -50

# Show Kubernetes manifests
ls k8s/
cat k8s/deployment.yaml | head -40

# Show smoke test script
head -50 scripts/smoke_test.sh
```

**Say:**
> "The CD pipeline automatically deploys to Kubernetes or Docker Compose. We have deployment manifests and smoke tests that run post-deployment to verify the service."

---

### **[3:15 - 4:15] M5: Live Prediction Demo**

**Show:** Terminal

**Commands to run:**
```bash
# Health check
curl -s http://localhost:8000/health | jq

# Make a prediction
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" | jq

# Check stats
curl -s http://localhost:8000/stats | jq

# Run load test (quick version)
for i in {1..5}; do 
  curl -s -X POST http://localhost:8000/predict -F "file=@test_image.jpg" > /dev/null
  echo "Request $i complete"
done
```

**Say:**
> "Now let's make a live prediction. I'll upload an image and the model will classify it as cat or dog. Watch the confidence score and inference time."

---

### **[4:15 - 4:50] M5: Monitoring Dashboard**

**Show:** Grafana (http://localhost:3000)

**Action:**
1. Login with admin/admin
2. Navigate to "Cats vs Dogs Classifier - ML Dashboard"
3. Show: Model Accuracy, Predictions by Class, Request Rate

**Say:**
> "Our Grafana dashboard shows real-time metrics including model accuracy, prediction distribution, request rates, and latency. You can see the requests we just made reflected here."

**Show:** Prometheus (http://localhost:9090)

**Action:** Query `model_predictions_total` or `http_requests_total`

---

### **[4:50 - 5:00] Conclusion**

**Show:** Terminal or README

**Say:**
> "This concludes our demonstration of the complete MLOps pipeline. We've shown model development with MLflow tracking, containerized deployment, automated CI/CD with GitHub Actions, and real-time monitoring with Prometheus and Grafana. Thank you for watching - Group 47."

---

## 🎯 Quick Commands for Demo

Copy-paste these commands during the demo:

```bash
# 1. Health check
curl -s http://localhost:8000/health | jq

# 2. Make prediction
curl -s -X POST http://localhost:8000/predict -F "file=@test_image.jpg" | jq

# 3. View stats
curl -s http://localhost:8000/stats | jq

# 4. Quick load test
for i in {1..5}; do curl -s -X POST http://localhost:8000/predict -F "file=@test_image.jpg" > /dev/null && echo "Request $i done"; done

# 5. Check metrics
curl -s http://localhost:8000/metrics | grep model_
```

---

## 🔗 URLs to Open

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | - |
| API Health | http://localhost:8000/health | - |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| MLflow | http://localhost:5001 | - |

---

## 📋 Key Points to Mention

1. **M1**: Git + DVC for versioning, MLflow for experiment tracking
2. **M2**: FastAPI REST API, Docker containerization
3. **M3**: GitHub Actions CI with testing, linting, and Docker build
4. **M4**: Kubernetes/Docker Compose deployment with smoke tests
5. **M5**: Prometheus metrics + Grafana dashboards, structured logging

---

## 🎥 Recording Tips

1. **Resolution**: Record at 1920x1080 or higher
2. **Font Size**: Increase terminal font to 14-16pt
3. **Speed**: Don't rush - speak clearly
4. **Zoom**: Zoom into important areas when needed
5. **Clean Desktop**: Hide unnecessary windows/notifications
6. **Practice**: Run through once before recording

Good luck with your recording! 🎬
