# 🚀 CÁC BƯỚC TIẾP THEO SAU KHI TRAIN MODEL

Bạn đã train xong 3 configs và có **model tốt nhất: efficientnet_b0_optimized (98.67% accuracy)**.

Dưới đây là các bước MLOps tiếp theo để deploy và sử dụng model.

---

## 📊 BƯỚC 1: ĐÁNH GIÁ CHI TIẾT MODEL (Đang chạy...)

```powershell
python src/evaluate.py --model-path models/efficientnet_b0_optimized/best_model.pth --val-dir validation
```

**Kết quả tạo ra:**
- `evaluation_results/confusion_matrix.png` - Ma trận nhầm lẫn
- `evaluation_results/per_class_accuracy.png` - Accuracy từng class
- `evaluation_results/classification_report.txt` - Precision, Recall, F1
- `evaluation_results/metrics.json` - Các metrics tổng hợp

**Để làm gì:**
- Phân tích class nào model dự đoán tốt/kém
- Tìm patterns sai (class nào thường bị nhầm với class nào)
- Quyết định có cần thu thập thêm data cho class yếu không

---

## 🔌 BƯỚC 2: TEST API PREDICTION

### 2.1. Start API Server

**Terminal 1** (PowerShell):
```powershell
python start_api.py
```

Hoặc với custom model:
```powershell
$env:MODEL_PATH="models/efficientnet_b0_optimized/best_model.pth"
$env:MODEL_NAME="efficientnet_b0"
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Kiểm tra:** Mở http://localhost:8000/docs (Swagger UI)

### 2.2. Test Predictions

**Terminal 2** (PowerShell):
```powershell
python test_api.py
```

Hoặc test thủ công với curl:
```powershell
# Health check
curl http://localhost:8000/health

# Predict single image
curl -X POST "http://localhost:8000/predict" -F "file=@validation/healthy/healthy_001.jpg"

# Model info
curl http://localhost:8000/model/info
```

**Kết quả:**
- Xem inference time (ms/request)
- Kiểm tra accuracy trên validation set
- Test với ảnh thật từ user

---

## 🐳 BƯỚC 3: DEPLOY VỚI DOCKER

### 3.1. Build Docker Image

```powershell
# Build API image
docker build -f docker/Dockerfile.api -t rice-disease-api:latest .

# Check image
docker images | Select-String rice-disease
```

### 3.2. Run Container

```powershell
# Run API container
docker run -d `
  --name rice-api `
  -p 8000:8000 `
  -v ${PWD}/models:/app/models `
  -e MODEL_PATH=/app/models/efficientnet_b0_optimized/best_model.pth `
  -e MODEL_NAME=efficientnet_b0 `
  rice-disease-api:latest

# Check logs
docker logs -f rice-api

# Test
curl http://localhost:8000/health
```

### 3.3. Docker Compose (Toàn bộ stack)

```powershell
# Start all services (API + MLflow + Airflow)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop all
docker-compose down
```

**Services:**
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080 (admin/admin)

---

## ⚙️ BƯỚC 4: SETUP AIRFLOW PIPELINES

### 4.1. Start Airflow

```powershell
# Initialize database (first time only)
docker-compose run airflow-webserver airflow db init

# Create admin user
docker-compose run airflow-webserver airflow users create `
  --username admin `
  --password admin `
  --firstname Admin `
  --lastname User `
  --role Admin `
  --email admin@example.com

# Start all Airflow services
docker-compose up -d
```

### 4.2. Access Airflow UI

Mở http://localhost:8080
- Username: `admin`
- Password: `admin`

### 4.3. Enable DAGs

**2 DAGs có sẵn:**

1. **`training_pipeline`** - Tự động retrain model
   - Schedule: Hàng tuần (Chủ Nhật 2AM)
   - Tasks:
     - Prepare data
     - Train model
     - Evaluate model
     - Compare với model cũ
     - Deploy nếu tốt hơn

2. **`deployment_pipeline`** - Deploy model mới
   - Trigger thủ công hoặc sau training
   - Tasks:
     - Validate model
     - Build Docker image
     - Deploy to production
     - Health check

**Trong UI:**
- Click toggle bên cạnh DAG name để enable
- Click DAG name → Trigger DAG để chạy thủ công

---

## 📈 BƯỚC 5: MONITORING VÀ MLFLOW

### 5.1. MLflow Model Registry

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Register best model
client = mlflow.tracking.MlflowClient()
model_uri = "runs://4b8e6057500b4b03bef452bac0c212dd/model"

# Create registered model
client.create_registered_model(
    name="rice-disease-classifier",
    description="Production model for rice disease classification"
)

# Add version
mv = client.create_model_version(
    name="rice-disease-classifier",
    source=model_uri,
    run_id="4b8e6057500b4b03bef452bac0c212dd"
)

# Promote to production
client.transition_model_version_stage(
    name="rice-disease-classifier",
    version=mv.version,
    stage="Production"
)
```

### 5.2. View Experiments

```powershell
# Xem tất cả runs
python src/view_all_results.py

# Tìm run cụ thể
python src/find_run.py efficientnet_b0_optimized_20251214_012804
```

**MLflow UI:** http://localhost:5000
- Compare runs
- Visualize metrics
- Download models
- Track experiments

### 5.3. Monitor API Performance

```python
# View Prometheus metrics
curl http://localhost:8000/metrics
```

**Metrics tracked:**
- `inference_requests_total` - Số request
- `inference_latency_seconds` - Latency
- `predictions_by_class` - Predictions per class

---

## 🧪 BƯỚC 6: A/B TESTING (Optional)

Deploy 2 models cùng lúc và so sánh:

```python
# deploy_ab_test.py
from fastapi import FastAPI
import random

app = FastAPI()

# Load 2 models
model_a = load_model("models/efficientnet_b0_optimized/best_model.pth")
model_b = load_model("models/mobilenetv3_large/best_model.pth")

@app.post("/predict")
async def predict(file: UploadFile):
    # Route 50% traffic to each model
    model = model_a if random.random() < 0.5 else model_b
    result = model.predict(file)
    result["model_version"] = "A" if model == model_a else "B"
    return result
```

**Track results:**
- So sánh accuracy trong production
- So sánh inference time
- Chọn model tốt hơn

---

## 🎯 BƯỚC 7: CONTINUOUS TRAINING

Setup tự động retrain khi có data mới:

1. **Thêm data mới vào `train/` và `validation/`**

2. **Trigger Airflow DAG:**
   ```powershell
   # Trigger via CLI
   docker-compose exec airflow-webserver airflow dags trigger training_pipeline

   # Hoặc trong Airflow UI
   ```

3. **Model tự động:**
   - Train với data mới
   - Evaluate và compare
   - Deploy nếu tốt hơn model cũ

---

## 📝 CHECKLIST ĐỂ DEPLOY LÊN PRODUCTION

- [ ] **Evaluation** - Xem confusion matrix, phân tích lỗi
- [ ] **API Testing** - Test với nhiều ảnh, measure latency
- [ ] **Docker** - Build và test container locally
- [ ] **MLflow Registry** - Register model với version
- [ ] **Airflow** - Setup automatic retraining schedule
- [ ] **Monitoring** - Setup alerts cho accuracy drop
- [ ] **Documentation** - Document API endpoints, model behavior
- [ ] **Security** - Add authentication, rate limiting
- [ ] **Cloud Deployment** - Deploy to AWS/GCP/Azure
- [ ] **CI/CD** - Setup GitHub Actions cho auto-deploy

---

## 🚀 QUICK START COMMANDS

```powershell
# 1. Evaluate model (đang chạy)
python src/evaluate.py --model-path models/efficientnet_b0_optimized/best_model.pth

# 2. Start API
python start_api.py

# 3. Test API (terminal khác)
python test_api.py

# 4. Docker
docker-compose up -d

# 5. View results
python src/view_all_results.py

# 6. MLflow UI
# Mở http://localhost:5000

# 7. Airflow UI
# Mở http://localhost:8080 (admin/admin)
```

---

Bạn muốn bắt đầu bước nào? Tôi sẽ hướng dẫn chi tiết!
