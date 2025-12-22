# 🚀 HƯỚNG DẪN DEMO HOÀN CHỈNH - MLOPS PROJECT
# Rice Leaf Disease Classification

> **Tài liệu demo tổng hợp tất cả các công cụ MLOps với các câu lệnh chính xác**

**Thời gian demo**: 30-45 phút  
**Ngày cập nhật**: December 15, 2025

---

## 📑 MỤC LỤC

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Chuẩn Bị Demo](#2-chuẩn-bị-demo)
3. [MLflow - Experiment Tracking](#3-mlflow---experiment-tracking--model-registry)
4. [Training Models](#4-training-models)
5. [FastAPI - REST API](#5-fastapi---rest-api-deployment)
6. [Docker - Containerization](#6-docker---containerization)
7. [Apache Airflow - Orchestration](#7-apache-airflow---workflow-orchestration)
8. [Prometheus & Grafana - Monitoring](#8-prometheus--grafana---monitoring)
9. [Demo Flow Hoàn Chỉnh](#9-demo-flow-hoàn-chỉnh)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. TỔNG QUAN HỆ THỐNG

### 🎯 Project Overview

**Rice Leaf Disease Classification** - Hệ thống MLOps End-to-End hoàn chỉnh

- **Model**: EfficientNet-B0 (98.67% accuracy)
- **Dataset**: 2,100 training + 528 validation images
- **Classes**: 6 loại bệnh lá lúa

### 🏗️ Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data → Training → MLflow → Model Registry → API → Monitor │
│    ↓        ↓         ↓           ↓           ↓       ↓    │
│  DVC    Airflow   Tracking   Versioning   FastAPI  Grafana │
│                                              Docker         │
│                                            Prometheus        │
└─────────────────────────────────────────────────────────────┘
```

### 📦 Services & Ports

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| MLflow | 5000 | http://localhost:5000 | - |
| FastAPI | 8000 | http://localhost:8000 | - |
| Swagger UI | 8000 | http://localhost:8000/docs | - |
| Airflow | 8080 | http://localhost:8080 | admin/admin |
| Prometheus | 9090 | http://localhost:9090 | - |
| Grafana | 3000 | http://localhost:3000 | admin/admin |
| PostgreSQL | 5432 | localhost:5432 | postgres/postgres |

---

## 2. CHUẨN BỊ DEMO

### 2.1. Kiểm Tra Environment

```powershell
# 1. Kiểm tra Python
python --version  # Expected: 3.9+

# 2. Kiểm tra Docker
docker --version
docker-compose --version

# 3. Kiểm tra Git
git --version

# 4. Navigate to project
cd e:\MLOps\Final\RiceLeafsDisease
```

### 2.2. Cài Đặt Dependencies (Nếu chưa có)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2.3. Kiểm Tra Dataset

```powershell
# Count training images
Get-ChildItem -Path .\train -Recurse -File | Measure-Object | Select-Object -ExpandProperty Count
# Expected: 2,100

# Count validation images
Get-ChildItem -Path .\validation -Recurse -File | Measure-Object | Select-Object -ExpandProperty Count
# Expected: 528

# List classes
Get-ChildItem -Path .\train -Directory | Select-Object Name
```

### 2.4. Start All Services (Quick Start)

```powershell
# Option 1: Start all with Docker Compose
docker-compose up -d

# Option 2: Start services individually
# (See sections below for detailed instructions)
```

### 2.5. Verify Services Running

```powershell
# Check Docker containers
docker-compose ps

# Check health endpoints
curl http://localhost:8000/health    # API
curl http://localhost:5000           # MLflow
curl http://localhost:8080           # Airflow
curl http://localhost:9090           # Prometheus
curl http://localhost:3000           # Grafana
```

---

## 3. MLFLOW - EXPERIMENT TRACKING & MODEL REGISTRY

### 3.1. Khởi Động MLflow Server

#### Option 1: Local (Simple)

```powershell
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Hoặc dùng UI mode
mlflow ui --port 5000
```

#### Option 2: With PostgreSQL Backend (Production)

```powershell
# Start PostgreSQL (if not running)
docker run -d `
  --name mlflow-postgres `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=mlflow `
  -p 5432:5432 `
  postgres:14

# Start MLflow with PostgreSQL
mlflow server `
  --backend-store-uri postgresql://postgres:postgres@localhost:5432/mlflow `
  --default-artifact-root ./mlruns `
  --host 0.0.0.0 `
  --port 5000
```

#### Option 3: Docker Compose (Recommended)

```powershell
# Already running if you did docker-compose up -d
# Access at http://localhost:5000
```

### 3.2. Verify MLflow Connection

```powershell
# Test connection
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print('✓ Connected:', mlflow.get_tracking_uri())"
```

### 3.3. MLflow UI - Key Features

**Open Browser**: http://localhost:5000

**Features to Demo**:

1. **Experiments Tab**
   - View all experiments
   - Filter by name, metrics

2. **Runs List**
   - See all training runs
   - Metrics: loss, accuracy, precision, recall, f1

3. **Compare Runs**
   - Select multiple runs → Compare
   - Side-by-side metrics comparison
   - Parallel coordinates plot

4. **Run Details**
   - Parameters logged
   - Metrics charts
   - Artifacts (model, plots)
   - System info

5. **Model Registry**
   - Registered models
   - Model versions
   - Stages: None → Staging → Production
   - Model lineage

### 3.4. MLflow CLI Commands

```powershell
# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0

# Describe run
mlflow runs describe --run-id <RUN_ID>

# List models
mlflow models list

# Serve model
mlflow models serve -m "models:/rice-disease-classifier/Production" -p 5001
```

---

## 4. TRAINING MODELS

### 4.1. Training Single Model

```powershell
# Basic training (50 epochs)
python src/train.py `
  --train-dir train `
  --val-dir validation `
  --epochs 50

# Training with custom parameters
python src/train.py `
  --train-dir train `
  --val-dir validation `
  --epochs 30 `
  --batch-size 32 `
  --lr 0.001 `
  --model-name efficientnet_b0

# Quick training for demo (5 epochs)
python src/train.py --epochs 5 --model-name efficientnet_b0
```

**What gets logged to MLflow**:
- ✅ All hyperparameters
- ✅ Loss, accuracy per epoch
- ✅ Precision, recall, f1-score
- ✅ Confusion matrix image
- ✅ Best model checkpoint
- ✅ Training time, GPU memory

### 4.2. Training Multiple Models (Comparison)

```powershell
# Compare 3 architectures
python src/train_comparison.py

# Or train individually
python src/train.py --model-name efficientnet_b0 --epochs 50
python src/train.py --model-name mobilenetv3_large --epochs 50
```

### 4.3. View Training Results

**In MLflow UI**:
1. Go to http://localhost:5000
2. Select experiment: `rice-disease-classification`
3. See all runs with metrics
4. Click run → View details
5. Compare multiple runs

**Via CLI**:

```powershell
# Find best run
python src/find_run.py

# View comparison results
python src/view_comparison.py

# View all results
python src/view_all_results.py
```

### 4.4. Evaluate Model

```powershell
# Evaluate on validation set
python src/evaluate.py `
  --val-dir validation `
  --model-path models/efficientnet_b0_optimized/best_model.pth `
  --model-name efficientnet_b0

# Results saved to: evaluation_results/metrics.json
```

### 4.5. Register Best Model

```powershell
# Setup model registry
python setup_model_registry.py

# Or register manually
python register_model.py
```

---

## 5. FASTAPI - REST API DEPLOYMENT

### 5.1. Start API Server

```powershell
# Method 1: Using start script (Recommended)
python start_api.py

# Method 2: Direct uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Docker (if using docker-compose)
docker-compose up -d api
```

**Expected Output**:
```
🚀 STARTING RICE DISEASE API
Model: models/efficientnet_b0_optimized/best_model.pth
Server: http://localhost:8000
Docs: http://localhost:8000/docs
```

### 5.2. API Endpoints Overview

**Interactive Docs**: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single image prediction |
| `/batch_predict` | POST | Batch predictions |
| `/metrics` | GET | Prometheus metrics |

### 5.3. Test API - Health Check

```powershell
# PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# curl (if available)
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 5.4. Test API - Single Prediction

#### Using Python Script

```powershell
# Quick test
python quick_test_api.py

# Detailed test
python test_api.py
```

#### Using PowerShell

```powershell
# Set variables
$imagePath = "validation\healthy\healthy_001.jpg"
$uri = "http://localhost:8000/predict"

# Send request
$form = @{
    file = Get-Item -Path $imagePath
}
$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form

# Display result
$response | ConvertTo-Json
```

#### Using Swagger UI

1. Open http://localhost:8000/docs
2. Find `/predict` endpoint
3. Click **"Try it out"**
4. Click **"Choose File"** → Select image
5. Click **"Execute"**
6. See response below

**Expected Response**:
```json
{
  "class_name": "healthy",
  "confidence": 0.9845,
  "probabilities": {
    "bacterial_leaf_blight": 0.0023,
    "brown_spot": 0.0098,
    "healthy": 0.9845,
    "leaf_blast": 0.0032,
    "leaf_scald": 0.0001,
    "narrow_brown_spot": 0.0001
  },
  "inference_time": 0.0234
}
```

### 5.5. Test API - Batch Prediction

```powershell
# Using curl (multiple files)
curl -X POST "http://localhost:8000/batch_predict" `
  -F "files=@validation/healthy/healthy_001.jpg" `
  -F "files=@validation/brown_spot/brown_spot_001.jpg" `
  -F "files=@validation/leaf_blast/leaf_blast_001.jpg"
```

### 5.6. Model Info

```powershell
curl http://localhost:8000/model/info
```

**Response**:
```json
{
  "model_name": "efficientnet_b0",
  "num_classes": 6,
  "total_parameters": 4049564,
  "trainable_parameters": 4007516,
  "device": "cuda",
  "class_names": [...]
}
```

---

## 6. DOCKER - CONTAINERIZATION

### 6.1. Docker Compose Overview

**File**: `docker-compose.yml`

**Services**:
- 🐘 PostgreSQL (database)
- 📊 MLflow (tracking server)
- 🤖 API (inference service)
- 🏋️ Trainer (training service)
- 🔄 Airflow Webserver
- 🔄 Airflow Scheduler
- 📈 Prometheus (monitoring)
- 📊 Grafana (visualization)

### 6.2. Build Docker Images

```powershell
# Build all images
docker-compose build

# Build specific service
docker build -f docker/Dockerfile.api -t rice-disease-api:latest .
docker build -f docker/Dockerfile.train -t rice-disease-trainer:latest .
docker build -f docker/Dockerfile.airflow -t rice-disease-airflow:latest .
```

### 6.3. Start All Services

```powershell
# Start all services in background
docker-compose up -d

# Start specific services
docker-compose up -d postgres mlflow api

# Start with logs
docker-compose up
```

**Expected Output**:
```
✔ Container rice-postgres       Started
✔ Container rice-mlflow         Started
✔ Container rice-api            Started
✔ Container rice-prometheus     Started
✔ Container rice-grafana        Started
✔ Container rice-airflow-webserver   Started
✔ Container rice-airflow-scheduler   Started
```

### 6.4. Check Services Status

```powershell
# List running containers
docker-compose ps

# Check specific service
docker ps | Select-String "rice-api"

# View resource usage
docker stats
```

### 6.5. View Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f airflow-webserver

# Last N lines
docker logs rice-api --tail 50
```

### 6.6. Exec into Containers

```powershell
# Access API container
docker exec -it rice-api bash

# Access MLflow container
docker exec -it rice-mlflow bash

# Access PostgreSQL
docker exec -it rice-postgres psql -U postgres -d mlflow
```

### 6.7. Manage Containers

```powershell
# Stop all services
docker-compose stop

# Start stopped services
docker-compose start

# Restart specific service
docker-compose restart api

# Stop and remove containers
docker-compose down

# Stop and remove with volumes
docker-compose down -v

# Remove everything (containers, images, volumes)
docker-compose down --rmi all -v
```

### 6.8. Test Services in Docker

```powershell
# Test API
curl http://localhost:8000/health

# Test MLflow
curl http://localhost:5000

# Test Prometheus
curl http://localhost:9090/-/healthy

# Test Grafana
curl http://localhost:3000/api/health
```

---

## 7. APACHE AIRFLOW - WORKFLOW ORCHESTRATION

### 7.1. Access Airflow UI

**URL**: http://localhost:8080  
**Username**: `admin`  
**Password**: `admin`

```powershell
# Open in browser
Start-Process http://localhost:8080
```

### 7.2. Available DAGs

#### DAG 1: Training Pipeline

**Name**: `rice_disease_training_pipeline`  
**Schedule**: `@weekly` (every Sunday)  
**Purpose**: Automated model training workflow

**Tasks**:
1. `validate_data` - Validate dataset availability
2. `setup_dvc` - Setup DVC for data versioning
3. `train_model` - Train model in Docker
4. `evaluate_model` - Evaluate model performance
5. `register_model` - Register to MLflow Registry
6. `notify_completion` - Send notification

**Task Dependencies**:
```
validate_data → setup_dvc → train_model → evaluate_model → register_model → notify_completion
```

#### DAG 2: Deployment Pipeline

**Name**: `rice_disease_deployment_pipeline`  
**Schedule**: Manual trigger  
**Purpose**: Deploy best model to production

**Tasks**:
1. `fetch_best_model` - Get best model from MLflow
2. `validate_model` - Validate model
3. `build_api_image` - Build Docker image
4. `deploy_api` - Deploy API container
5. `health_check` - Verify deployment
6. `notify_deployment` - Send notification

### 7.3. Enable DAG

```powershell
# Enable via CLI
docker exec rice-airflow-webserver airflow dags unpause rice_disease_training_pipeline

# Or use UI: Toggle switch next to DAG name
```

### 7.4. Trigger DAG

#### Method 1: UI

1. Find DAG in list
2. Click **▶️ Play** button (right side)
3. Click **"Trigger DAG"**
4. (Optional) Add config JSON
5. Click **"Trigger"**

#### Method 2: CLI

```powershell
# Trigger training pipeline
docker exec rice-airflow-webserver airflow dags trigger rice_disease_training_pipeline

# Trigger with config
docker exec rice-airflow-webserver airflow dags trigger rice_disease_training_pipeline --conf '{"epochs": 10}'

# Trigger deployment pipeline
docker exec rice-airflow-webserver airflow dags trigger rice_disease_deployment_pipeline
```

### 7.5. Monitor DAG Execution

#### View Graph

1. Click on DAG name
2. See **Graph View** (default)
3. Watch tasks change colors:
   - ⚪ Queued (white)
   - 🟡 Running (yellow)
   - 🟢 Success (green)
   - 🔴 Failed (red)

#### View Logs

1. Click on a task box
2. Click **"Log"** button
3. View realtime execution logs
4. Check for errors/warnings

#### Other Views

- **Grid**: Historical runs in grid format
- **Calendar**: Runs on calendar view
- **Code**: View DAG source code
- **Gantt**: Task timeline

### 7.6. Airflow CLI Commands

```powershell
# List all DAGs
docker exec rice-airflow-webserver airflow dags list

# List tasks in DAG
docker exec rice-airflow-webserver airflow tasks list rice_disease_training_pipeline

# View DAG structure
docker exec rice-airflow-webserver airflow dags show rice_disease_training_pipeline

# Test single task (without running full DAG)
docker exec rice-airflow-webserver airflow tasks test rice_disease_training_pipeline validate_data 2024-01-01

# View task logs
docker exec rice-airflow-webserver airflow tasks logs rice_disease_training_pipeline train_model <execution_date> 1

# List recent runs
docker exec rice-airflow-webserver airflow dags list-runs -d rice_disease_training_pipeline

# Clear failed tasks (to retry)
docker exec rice-airflow-webserver airflow tasks clear rice_disease_training_pipeline --task-regex "train_model" -y
```

### 7.7. Check DAG Status

```powershell
# Get DAG state
docker exec rice-airflow-webserver airflow dags state rice_disease_training_pipeline <execution_date>

# List task instances
docker exec rice-airflow-webserver airflow tasks states-for-dag-run rice_disease_training_pipeline <execution_date>
```

---

## 8. PROMETHEUS & GRAFANA - MONITORING

### 8.1. Generate Traffic for Metrics

```powershell
# Generate 20 requests
python generate_traffic.py
# Choose option 1

# Generate 100 requests automatically
python generate_demo_traffic.py

# Check current metrics
python test_metrics.py
```

### 8.2. Prometheus

**URL**: http://localhost:9090

```powershell
# Open Prometheus
Start-Process http://localhost:9090
```

#### Key Metrics Available

| Metric | Description |
|--------|-------------|
| `inference_requests_total` | Total API requests |
| `inference_latency_seconds` | Request latency histogram |
| `predictions_by_class_total` | Predictions per class |
| `process_cpu_seconds_total` | CPU usage |
| `process_resident_memory_bytes` | Memory usage |

#### Useful Queries

```promql
# 1. Total requests
sum(inference_requests_total)

# 2. Request rate (per second)
rate(inference_requests_total[1m])

# 3. Requests in last 5 minutes
increase(inference_requests_total[5m])

# 4. Average latency
rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])

# 5. 95th percentile latency
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))

# 6. Top 3 predicted classes
topk(3, predictions_by_class_total)

# 7. Predictions by class
sum by (class_name) (predictions_by_class_total)

# 8. Memory usage (MB)
process_resident_memory_bytes / 1024 / 1024

# 9. CPU time
rate(process_cpu_seconds_total[5m])
```

#### How to Query

1. Go to http://localhost:9090
2. Enter query in **Expression** box
3. Click **Execute**
4. Switch between **Table** and **Graph** views
5. Adjust time range (e.g., Last 15 minutes)

### 8.3. Grafana

**URL**: http://localhost:3000  
**Username**: `admin`  
**Password**: `admin`

```powershell
# Open Grafana
Start-Process http://localhost:3000
```

#### Setup Data Source

1. Login with `admin/admin`
2. (First time) Skip password change or set new password
3. Click **⚙️ Configuration** → **Data sources**
4. Click **Add data source**
5. Select **Prometheus**
6. Configure:
   - **Name**: Prometheus
   - **URL**: `http://prometheus:9090` (for Docker)
   - Or `http://localhost:9090` (for local)
7. Click **Save & Test**
8. Should see: ✅ "Data source is working"

#### Create Dashboard

**Option 1: Quick Single Panel**

1. Click **+** → **Dashboard**
2. Click **Add new panel**
3. In query editor:
   - Select **Prometheus** datasource
   - Enter query: `inference_requests_total`
4. Click **Run query**
5. Set panel title: "Total API Requests"
6. Select visualization type: **Stat**
7. Click **Apply**
8. Click **💾 Save dashboard**

**Option 2: Complete Dashboard**

Create multiple panels:

**Panel 1: Total Requests**
- Query: `inference_requests_total`
- Visualization: Stat
- Title: "Total Requests"

**Panel 2: Request Rate**
- Query: `rate(inference_requests_total[1m])`
- Visualization: Graph
- Title: "Request Rate (req/s)"
- Unit: "ops/s"

**Panel 3: Average Latency**
- Query: `rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])`
- Visualization: Graph
- Title: "Average Response Time"
- Unit: "seconds (s)"

**Panel 4: P95 Latency**
- Query: `histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))`
- Visualization: Graph
- Title: "95th Percentile Latency"

**Panel 5: Predictions Distribution**
- Query: `predictions_by_class_total`
- Visualization: Bar chart or Pie chart
- Title: "Predictions by Class"
- Legend: `{{class_name}}`

**Panel 6: Memory Usage**
- Query: `process_resident_memory_bytes / 1024 / 1024`
- Visualization: Graph
- Title: "Memory Usage (MB)"

#### Dashboard Settings

- **Auto-refresh**: Set to 5s or 10s
- **Time range**: Last 15 minutes or Last 1 hour
- **Theme**: Dark or Light

#### Save Dashboard

1. Click **💾 Save dashboard** (top right)
2. Enter name: "Rice Disease API Monitoring"
3. Click **Save**

### 8.4. Real-time Monitoring Demo

```powershell
# Terminal 1: Generate continuous traffic
while($true) { 
    python quick_test_api.py
    Start-Sleep -Seconds 2
}

# Terminal 2: Watch metrics
python test_metrics.py

# Browser: Watch Grafana dashboard update in realtime
```

---

## 9. DEMO FLOW HOÀN CHỈNH

### 🎬 Demo Script (30-45 phút)

#### Phần 1: Giới Thiệu & Setup (5 phút)

```powershell
# 1. Navigate to project
cd e:\MLOps\Final\RiceLeafsDisease

# 2. Check dataset
Get-ChildItem train/ -Directory
Get-ChildItem validation/ -Directory

# 3. Show project structure
tree /F /A
```

**Talking Points**:
- Giới thiệu bài toán: 6 loại bệnh lá lúa
- Dataset: 2,100 train / 528 validation
- Architecture: EfficientNet-B0
- MLOps tools được sử dụng

#### Phần 2: MLflow Tracking (5 phút)

```powershell
# 1. Start MLflow (nếu chưa chạy)
mlflow server --host 0.0.0.0 --port 5000

# 2. Open UI
Start-Process http://localhost:5000

# 3. Train quick model (demo)
python src/train.py --epochs 5 --model-name efficientnet_b0
```

**Demo trong UI**:
- Xem experiments list
- Click vào run mới nhất
- Show parameters logged
- Show metrics charts (loss, accuracy)
- Show artifacts (model file, confusion matrix)

**Talking Points**:
- MLflow tracking tự động log parameters, metrics
- Compare multiple runs easily
- Artifact storage cho reproducibility

#### Phần 3: FastAPI Deployment (8 phút)

```powershell
# 1. Start API
python start_api.py

# 2. Open Swagger docs
Start-Process http://localhost:8000/docs

# 3. Test health
curl http://localhost:8000/health

# 4. Test prediction
python quick_test_api.py
```

**Demo trong Swagger UI**:
1. Mở http://localhost:8000/docs
2. Thử `/health` endpoint
3. Thử `/model/info` endpoint
4. Thử `/predict` endpoint:
   - Click "Try it out"
   - Upload image từ validation/healthy/
   - Execute và xem response
   - Show confidence scores

**Talking Points**:
- FastAPI cung cấp automatic API docs
- Input validation với Pydantic
- Async support cho high performance
- Easy to test với interactive UI

#### Phần 4: Docker Orchestration (7 phút)

```powershell
# 1. Show docker-compose.yml
code docker-compose.yml

# 2. Start all services
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. Test services
curl http://localhost:8000/health  # API
curl http://localhost:5000         # MLflow
curl http://localhost:8080         # Airflow
```

**Demo**:
- Show 8 services running
- Explain each service purpose
- Show logs: `docker-compose logs -f api`
- Show resource usage: `docker stats`

**Talking Points**:
- Docker containerization cho consistency
- docker-compose orchestrates multiple services
- Easy to deploy anywhere
- Reproducible environment

#### Phần 5: Airflow Orchestration (8 phút)

```powershell
# 1. Open Airflow UI
Start-Process http://localhost:8080

# 2. Login (admin/admin)

# 3. Enable DAG
docker exec rice-airflow-webserver airflow dags unpause rice_disease_training_pipeline

# 4. Trigger pipeline
docker exec rice-airflow-webserver airflow dags trigger rice_disease_training_pipeline
```

**Demo trong UI**:
1. Show DAGs list
2. Click vào training pipeline
3. Show Graph View:
   - Explain task dependencies
   - Show workflow visualization
4. Trigger DAG và watch execution
5. Click vào task → View logs
6. Show Grid view (historical runs)

**Talking Points**:
- Airflow orchestrates complex workflows
- Automatic scheduling (@weekly)
- Retry mechanism cho failed tasks
- Monitoring và logging built-in
- Integration với Docker, MLflow

#### Phần 6: Monitoring với Prometheus & Grafana (7 phút)

```powershell
# 1. Generate traffic
python generate_demo_traffic.py

# 2. Open Prometheus
Start-Process http://localhost:9090

# 3. Open Grafana
Start-Process http://localhost:3000
```

**Demo Prometheus**:
1. Go to Graph tab
2. Query: `inference_requests_total`
3. Query: `rate(inference_requests_total[1m])`
4. Query: `predictions_by_class_total`
5. Show graph visualization

**Demo Grafana**:
1. Login (admin/admin)
2. Add Prometheus datasource
3. Create dashboard
4. Add panel với query: `inference_requests_total`
5. Add graph panel với query: `rate(inference_requests_total[1m])`
6. Show realtime updates

**Generate more traffic**:
```powershell
# Terminal mới
python generate_demo_traffic.py
# Watch dashboard update realtime
```

**Talking Points**:
- Prometheus scrapes metrics từ API
- Grafana visualizes metrics beautifully
- Real-time monitoring
- Alerts có thể setup cho anomalies
- Production-ready monitoring stack

---

## 10. TROUBLESHOOTING

### 10.1. MLflow Issues

#### MLflow không start

```powershell
# Check port
netstat -ano | findstr :5000

# Kill process if needed
taskkill /PID <PID> /F

# Restart
mlflow server --host 0.0.0.0 --port 5000
```

#### Connection refused

```powershell
# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Set correct URI
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

### 10.2. API Issues

#### API không start

```powershell
# Check model file exists
Test-Path models/efficientnet_b0_optimized/best_model.pth

# Check port
netstat -ano | findstr :8000

# Restart with debug
uvicorn api.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

#### Prediction fails

```powershell
# Verify image file
python -c "from PIL import Image; img = Image.open('validation/healthy/healthy_001.jpg'); print(img.size)"

# Test with curl
curl -X POST "http://localhost:8000/predict" -F "file=@validation/healthy/healthy_001.jpg"
```

### 10.3. Docker Issues

#### Containers won't start

```powershell
# View logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d

# Check disk space
docker system df
```

#### Port conflicts

```powershell
# Find process using port
netstat -ano | findstr :<PORT>

# Kill process
taskkill /PID <PID> /F
```

#### Build errors

```powershell
# Clean build
docker-compose build --no-cache

# Prune system
docker system prune -a
```

### 10.4. Airflow Issues

#### DAGs not showing

```powershell
# Check DAG errors
docker exec rice-airflow-webserver airflow dags list-import-errors

# Verify DAG syntax
python airflow/dags/training_pipeline.py

# Restart scheduler
docker-compose restart airflow-scheduler
```

#### Tasks fail

```powershell
# View logs
docker exec rice-airflow-webserver airflow tasks logs <dag_id> <task_id> <execution_date>

# Clear to retry
docker exec rice-airflow-webserver airflow tasks clear <dag_id> --task-id <task_id> -y
```

### 10.5. Prometheus & Grafana Issues

#### Prometheus no data

```powershell
# Check targets: http://localhost:9090/targets
# Should show rice-api as UP

# Check metrics endpoint
curl http://localhost:8000/metrics

# Restart Prometheus
docker-compose restart prometheus
```

#### Grafana can't connect

```powershell
# Check datasource URL
# Docker: http://prometheus:9090
# Local: http://localhost:9090

# Test connection in datasource settings
# Click "Save & Test"
```

---

## 📚 ADDITIONAL RESOURCES

### Quick Commands Reference

```powershell
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart <service>

# Rebuild and restart
docker-compose up -d --build <service>

# Check status
docker-compose ps
```

### Useful Scripts

```powershell
# Generate traffic
python generate_demo_traffic.py

# Test API
python quick_test_api.py
python test_api.py

# Test metrics
python test_metrics.py

# Find best model
python src/find_run.py

# View results
python src/view_all_results.py
```

### Documentation Files

- [DEMO_COMPLETE.md](DEMO_COMPLETE.md) - Chi tiết từng tool
- [PROMETHEUS_GRAFANA_DEMO.md](PROMETHEUS_GRAFANA_DEMO.md) - Monitoring guide
- [AIRFLOW_DEMO_GUIDE.md](AIRFLOW_DEMO_GUIDE.md) - Airflow detailed guide
- [README.md](README.md) - Project overview
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Architecture details

### URLs Quick Access

```powershell
# Open all UIs
Start-Process http://localhost:5000   # MLflow
Start-Process http://localhost:8000   # API
Start-Process http://localhost:8000/docs  # Swagger
Start-Process http://localhost:8080   # Airflow
Start-Process http://localhost:9090   # Prometheus
Start-Process http://localhost:3000   # Grafana
```

---

## 🎯 DEMO CHECKLIST

### Pre-Demo Setup

- [ ] All dependencies installed
- [ ] Dataset downloaded (2,100 + 528 images)
- [ ] Docker Desktop running
- [ ] All containers started
- [ ] Model trained (at least 1 run)

### During Demo

**MLflow**:
- [ ] MLflow UI accessible
- [ ] Show experiment tracking
- [ ] Show model comparison
- [ ] Show model registry

**API**:
- [ ] API running and healthy
- [ ] Swagger UI working
- [ ] Successful prediction demo
- [ ] Show different disease classes

**Docker**:
- [ ] All 8 containers running
- [ ] Show docker-compose.yml
- [ ] Show logs
- [ ] Show resource usage

**Airflow**:
- [ ] Airflow UI accessible
- [ ] DAG visible and enabled
- [ ] Trigger pipeline
- [ ] Show graph execution
- [ ] Show task logs

**Monitoring**:
- [ ] Generate traffic (100+ requests)
- [ ] Prometheus queries working
- [ ] Grafana dashboard created
- [ ] Realtime updates visible

### Post-Demo

- [ ] Save dashboard configurations
- [ ] Export MLflow runs
- [ ] Stop containers (if needed)
- [ ] Clean up resources

---

## 🚀 QUICK START FOR DEMO

```powershell
# 1. Navigate to project
cd e:\MLOps\Final\RiceLeafsDisease

# 2. Start all services
docker-compose up -d

# 3. Wait for services to be ready (30 seconds)
Start-Sleep -Seconds 30

# 4. Generate traffic for monitoring
python generate_demo_traffic.py

# 5. Open all UIs
Start-Process http://localhost:5000   # MLflow
Start-Process http://localhost:8000/docs  # API Docs
Start-Process http://localhost:8080   # Airflow
Start-Process http://localhost:9090   # Prometheus
Start-Process http://localhost:3000   # Grafana

# 6. Trigger Airflow pipeline
docker exec rice-airflow-webserver airflow dags unpause rice_disease_training_pipeline
docker exec rice-airflow-webserver airflow dags trigger rice_disease_training_pipeline

# Now you're ready to demo! 🎉
```

---

**🎉 CHÚC BẠN DEMO THÀNH CÔNG!**

*Made with ❤️ for MLOps Education*
