# 📊 Slide Structure - Rice Leaf Disease Classification MLOps Project

**Total: 28-30 slides** | Duration: 20-25 phút presentation

---

## 🎯 SECTION 1: OPENING (3 slides)

### Slide 1: Title Slide
**Content:**
- **Project Title**: Rice Leaf Disease Classification with End-to-End MLOps Pipeline
- **Subtitle**: Production-Ready ML System with Monitoring & Automation
- Team members & roles
- University/Institution
- Date

**Visual:** Professional title page với icon/image của rice plant

---

### Slide 2: Agenda / Table of Contents
**Content:**
```
1. Introduction & Problem Statement
2. Dataset Overview
3. System Architecture
4. ML Pipeline & Training
5. MLOps Implementation
6. Tools & Technologies
7. Results & Evaluation
8. Live Demo
9. Conclusion & Future Work
```

**Visual:** Numbered list với icons cho mỗi section

---

### Slide 3: Project Overview
**Content:**
- **What**: Automated ML system để phát hiện 6 loại bệnh lúa
- **Why**: Giúp nông dân phát hiện sớm, tăng năng suất
- **How**: End-to-end MLOps pipeline từ training đến production monitoring
- **Impact**: 95% accuracy, real-time inference <25ms

**Visual:**
- Infographic: Problem → Solution → Impact
- Hoặc: Before vs After (manual detection vs automated)

---

## 🎯 SECTION 2: INTRODUCTION & PROBLEM (3 slides)

### Slide 4: Problem Statement
**Content:**
- **Current Challenge**:
  - Rice diseases gây mất 30-40% năng suất mỗi năm
  - Phát hiện manual: slow, requires expertise
  - Misdiagnosis → wrong treatment → crop loss

- **Our Solution**:
  - AI-powered disease classification
  - Real-time detection via mobile/web app
  - Automated ML pipeline ensure model quality

**Visual:**
- Images: Diseased rice leaves
- Statistics chart: Crop loss due to diseases
- Icon comparison: Manual vs Automated

---

### Slide 5: Project Objectives
**Content:**
**Technical Objectives:**
- ✅ Build deep learning model với accuracy >90%
- ✅ Implement end-to-end MLOps pipeline
- ✅ Deploy production-ready REST API
- ✅ Setup monitoring & alerting system
- ✅ Automate training & deployment

**Business Objectives:**
- 🎯 Fast detection (<30ms inference)
- 🎯 Scalable system (100+ req/s)
- 🎯 Easy to maintain & update
- 🎯 Low operational cost

**Visual:** Checklist với icons, hoặc split screen Technical vs Business

---

### Slide 6: Project Scope
**Content:**
**In Scope:**
- 6 rice disease classes classification
- EfficientNet-based model architecture
- MLflow experiment tracking
- Docker containerization
- CI/CD pipeline with Airflow
- Prometheus + Grafana monitoring
- REST API with FastAPI

**Out of Scope:**
- Disease severity estimation
- Treatment recommendation
- Mobile app development (future work)
- Multi-crop support

**Visual:** Venn diagram hoặc In/Out table

---

## 🎯 SECTION 3: DATASET (3 slides)

### Slide 7: Dataset Overview
**Content:**
- **Source**: Rice Leaf Disease Image Dataset
- **Total Images**: ~3,600 images
- **Split**:
  - Training: ~3,000 (80%)
  - Validation: ~600 (20%)
- **Image Size**: 224x224 RGB
- **Format**: JPEG

**6 Classes:**
1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy
4. Leaf Blast
5. Leaf Scald
6. Narrow Brown Spot

**Visual:**
- Pie chart showing class distribution
- Grid of sample images (1 per class)

---

### Slide 8: Data Statistics
**Content:**
**Class Distribution:**
| Class | Training | Validation | Total |
|-------|----------|------------|-------|
| Bacterial Leaf Blight | 500 | 100 | 600 |
| Brown Spot | 500 | 100 | 600 |
| Healthy | 500 | 100 | 600 |
| Leaf Blast | 500 | 100 | 600 |
| Leaf Scald | 500 | 100 | 600 |
| Narrow Brown Spot | 500 | 100 | 600 |
| **Total** | **3,000** | **600** | **3,600** |

**Key Points:**
- ✅ Balanced dataset (equal samples per class)
- ✅ No significant class imbalance
- ✅ High quality images with clear symptoms

**Visual:**
- Bar chart: Samples per class
- Before/After augmentation comparison

---

### Slide 9: Data Preprocessing & Augmentation
**Content:**
**Preprocessing Steps:**
1. Resize to 224x224
2. Normalize (ImageNet mean/std)
3. Convert to Tensor

**Data Augmentation (Training):**
- ✅ Random Rotation (±15°)
- ✅ Random Horizontal Flip
- ✅ Color Jitter (brightness, contrast, saturation)
- ✅ Random Crop & Resize
- ✅ Normalization

**Why Augmentation?**
- Increase dataset diversity
- Improve model generalization
- Reduce overfitting

**Visual:**
- Side-by-side: Original image → Augmented variations
- Flow diagram: Image → Preprocessing → Augmentation → Model

---

## 🎯 SECTION 4: ARCHITECTURE (4 slides)

### Slide 10: System Architecture Overview
**Content:**
**High-Level Architecture:**

```
┌─────────────┐
│   Data      │
│  Training   │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Model     │────▶ │   MLflow     │
│  Training   │      │  Tracking    │
└──────┬──────┘      └──────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Model     │────▶ │  Airflow     │
│  Registry   │      │ Orchestration│
└──────┬──────┘      └──────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│  FastAPI    │────▶ │ Prometheus   │
│  Inference  │      │ + Grafana    │
└─────────────┘      └──────────────┘
```

**Layers:**
1. Data Layer
2. Training Layer
3. Orchestration Layer
4. Serving Layer
5. Monitoring Layer

**Visual:** Architecture diagram với colors và icons (use Mermaid hoặc draw.io)

---

### Slide 11: Model Architecture
**Content:**
**EfficientNet-B0 Architecture:**

```
Input (224x224x3)
       ↓
┌─────────────────┐
│ EfficientNet-B0 │ ← Pretrained on ImageNet
│   Backbone      │   (4M parameters)
└────────┬────────┘
         │
    ┌────▼────┐
    │ Dropout │ (0.3)
    └────┬────┘
         │
    ┌────▼────┐
    │  Dense  │ (6 classes)
    └────┬────┘
         │
    ┌────▼────┐
    │ Softmax │
    └─────────┘
     Output
```

**Why EfficientNet-B0?**
- ✅ SOTA accuracy với low parameters
- ✅ Fast inference (~20ms)
- ✅ Good balance: accuracy vs speed
- ✅ Transfer learning từ ImageNet

**Visual:**
- Architecture diagram
- Comparison table: EfficientNet vs ResNet vs MobileNet

---

### Slide 12: MLOps Architecture
**Content:**
**Complete MLOps Pipeline:**

```
┌──────────────────────────────────────────┐
│          Data & Training                 │
│  • PyTorch + timm (EfficientNet)         │
│  • Data augmentation pipeline            │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│     Experiment Tracking (MLflow)         │
│  • Track metrics, params, artifacts      │
│  • Model versioning & registry           │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│     Orchestration (Airflow)              │
│  • Training pipeline (weekly)            │
│  • Deployment pipeline (on-demand)       │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│     Deployment (Docker + FastAPI)        │
│  • REST API (Swagger docs)               │
│  • Health checks & auto-restart          │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│     Monitoring (Prometheus + Grafana)    │
│  • Real-time metrics & dashboards        │
│  • Alerting on anomalies                 │
└──────────────────────────────────────────┘
```

**Visual:** Layered architecture với arrows showing data flow

---

### Slide 13: Technology Stack
**Content:**
**ML & DL:**
- PyTorch 2.0+
- timm (EfficientNet)
- torchvision
- scikit-learn

**MLOps Tools:**
- MLflow (tracking & registry)
- Apache Airflow (orchestration)
- Docker & Docker Compose
- Prometheus + Grafana

**API & Web:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Swagger/OpenAPI

**Infrastructure:**
- PostgreSQL (metadata)
- Docker Networks
- Volume persistence

**Testing:**
- Pytest
- Coverage.py

**Visual:**
- Icons grid của các tools
- Hoặc: Tech stack pyramid

---

## 🎯 SECTION 5: TRAINING PIPELINE (4 slides)

### Slide 14: Training Configuration
**Content:**
**Hyperparameters:**
```
Model: EfficientNet-B0
Optimizer: AdamW
Learning Rate: 1e-4 (with cosine annealing)
Batch Size: 32
Epochs: 50
Weight Decay: 1e-4
Dropout: 0.3
Mixed Precision: True (AMP)
```

**Training Strategy:**
- Transfer learning (ImageNet pretrained)
- Fine-tune all layers
- Early stopping (patience=10)
- Model checkpoint (save best only)
- Learning rate warmup (5 epochs)

**Visual:**
- Table of hyperparameters
- Training curve preview

---

### Slide 15: Training Process
**Content:**
**Training Pipeline Steps:**

```
1. Data Loading
   ├─ Load images from train/
   ├─ Apply augmentation
   └─ Create DataLoader (batch=32)

2. Model Initialization
   ├─ Load EfficientNet-B0
   ├─ Add custom classifier head
   └─ Move to GPU

3. Training Loop
   ├─ Forward pass
   ├─ Compute loss (CrossEntropy)
   ├─ Backward pass
   ├─ Update weights
   └─ Log to MLflow

4. Validation
   ├─ Evaluate on val set
   ├─ Compute metrics (acc, f1)
   └─ Save best checkpoint

5. Logging
   ├─ MLflow tracking
   ├─ Save artifacts (model, plots)
   └─ Register to model registry
```

**Visual:** Flowchart hoặc pipeline diagram

---

### Slide 16: Loss Functions & Metrics
**Content:**
**Loss Function:**
- Cross-Entropy Loss
- Formula: $L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$

**Evaluation Metrics:**
1. **Accuracy**: Overall correctness
   - $\text{Accuracy} = \frac{TP + TN}{Total}$

2. **Precision**: Positive prediction accuracy
   - $\text{Precision} = \frac{TP}{TP + FP}$

3. **Recall**: Actual positive detection rate
   - $\text{Recall} = \frac{TP}{TP + FN}$

4. **F1-Score**: Harmonic mean of precision & recall
   - $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Why Multiple Metrics?**
- Accuracy alone không đủ
- F1 score balance precision & recall
- Important cho medical/agricultural AI

**Visual:**
- Confusion matrix example
- Metric formulas với diagrams

---

### Slide 17: Model Optimization
**Content:**
**Optimization Techniques:**

1. **Mixed Precision Training (AMP)**
   - 40% faster training
   - 50% less GPU memory
   - No accuracy loss

2. **Learning Rate Scheduling**
   - Cosine Annealing
   - Warmup first 5 epochs
   - Better convergence

3. **Data Augmentation**
   - Reduce overfitting
   - Improve generalization

4. **Early Stopping**
   - Stop when val loss plateau
   - Save best model only

5. **Weight Decay (L2 Regularization)**
   - Prevent overfitting
   - λ = 1e-4

**Results:**
- Training time: 2 hours (50 epochs)
- Final val accuracy: 95.08%
- Model size: 16MB

**Visual:**
- Before/After optimization comparison
- Training curves showing convergence

---

## 🎯 SECTION 6: MLOPS IMPLEMENTATION (7 slides)

### Slide 18: Experiment Tracking với MLflow
**Content:**
**MLflow Components:**

1. **Tracking Server**
   - Log parameters, metrics, artifacts
   - Web UI: http://localhost:5000
   - PostgreSQL backend

2. **What We Track:**
   ```python
   mlflow.log_params({
       "model": "efficientnet_b0",
       "lr": 1e-4,
       "batch_size": 32,
       "epochs": 50
   })

   mlflow.log_metrics({
       "train_acc": 0.98,
       "val_acc": 0.95,
       "val_loss": 0.15
   })

   mlflow.log_artifact("confusion_matrix.png")
   ```

3. **Benefits:**
   - Compare 10+ experiments easily
   - Reproducibility (track everything)
   - Collaboration (shared tracking server)

**Visual:**
- MLflow UI screenshot
- Comparison chart của multiple runs

---

### Slide 19: Model Registry & Versioning
**Content:**
**Model Registry Workflow:**

```
Train Model
    ↓
Log to MLflow
    ↓
Register to Registry
    ↓
┌─────────────┐
│   Version   │
│   Control   │
└──────┬──────┘
       │
   ┌───┴────┬────────┐
   ▼        ▼        ▼
  None  Staging  Production
   │        │        │
   │     Test &   Deploy to
   │    Validate   Production
   │        │
   └────────▼
      Archived
```

**Model Stages:**
- **None**: Newly registered
- **Staging**: Testing environment
- **Production**: Live serving
- **Archived**: Old versions

**Benefits:**
- Version control cho models
- Easy rollback nếu có issue
- Traceability (which model deployed?)

**Visual:**
- State machine diagram
- MLflow Registry screenshot

---

### Slide 20: CI/CD Pipeline
**Content:**
**Automated Pipeline:**

```
Code Push (git push)
       ↓
┌──────────────┐
│    Tests     │ ← pytest, coverage
└──────┬───────┘
       ↓ (pass)
┌──────────────┐
│   Linting    │ ← flake8, black
└──────┬───────┘
       ↓ (pass)
┌──────────────┐
│   Security   │ ← bandit scan
└──────┬───────┘
       ↓ (pass)
┌──────────────┐
│ Build Docker │ ← docker build
└──────┬───────┘
       ↓
┌──────────────┐
│Push Registry │ ← Tag: v1.0.0
└──────┬───────┘
       ↓
┌──────────────┐
│Deploy Staging│ ← Smoke tests
└──────┬───────┘
       ↓ (pass)
┌──────────────┐
│Deploy Prod   │ ← Gradual rollout
└──────────────┘
```

**Key Features:**
- ✅ Automated testing (block bad code)
- ✅ Docker image versioning
- ✅ Staging → Production flow
- ✅ Rollback capability

**Visual:** Pipeline flowchart với colors (green=pass, red=fail)

---

### Slide 21: Orchestration với Airflow
**Content:**
**Two Main Pipelines:**

**1. Training Pipeline (Weekly):**
```
validate_data → setup_dvc → train_model
→ evaluate_model → notify_completion
```
- Schedule: Every Monday 2AM
- Auto-retry on failure (3x)
- Email notification

**2. Deployment Pipeline (On-Demand):**
```
validate_model → build_docker_image
→ deploy_staging → smoke_tests
→ deploy_production
```
- Triggered khi có model mới
- Quality gate: accuracy > 80%
- Automated testing

**Why Airflow?**
- ✅ Dependency management
- ✅ Retry logic
- ✅ Web UI monitoring
- ✅ Scalable (Kubernetes executor)

**Visual:**
- Airflow DAG Graph View screenshots
- Gantt chart showing task durations

---

### Slide 22: API Deployment với FastAPI
**Content:**
**REST API Endpoints:**

```python
GET  /                  # Root info
GET  /health            # Health check
GET  /model/info        # Model metadata
POST /predict           # Inference
GET  /metrics           # Prometheus metrics
```

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_image.jpg"
```

**Sample Response:**
```json
{
  "class_name": "leaf_blast",
  "confidence": 0.9823,
  "probabilities": {
    "leaf_blast": 0.9823,
    "brown_spot": 0.0098,
    ...
  },
  "inference_time": 0.0234
}
```

**Features:**
- ✅ Swagger UI docs (auto-generated)
- ✅ Input validation
- ✅ Error handling
- ✅ Prometheus metrics integration
- ✅ Health checks

**Visual:**
- Swagger UI screenshot
- API flow diagram

---

### Slide 23: Monitoring với Prometheus & Grafana
**Content:**
**Prometheus Metrics:**
```
# Counter
inference_requests_total

# Histogram
inference_latency_seconds

# Counter by label
predictions_by_class_total{class="leaf_blast"}
```

**Grafana Dashboard (7 panels):**
1. Request Rate (req/s)
2. Average Response Time (ms)
3. P95 Latency
4. Total Requests
5. Predictions by Class (bar chart)
6. Request Count Over Time
7. System Health (%)

**Alerting Rules:**
- Response time > 100ms
- Error rate > 5%
- Prediction confidence < 70%

**Benefits:**
- Real-time monitoring
- Early issue detection
- Performance optimization insights

**Visual:**
- Grafana dashboard screenshot
- Metrics trend chart

---

### Slide 24: Containerization với Docker
**Content:**
**Docker Architecture:**

```
docker-compose.yml
├─ postgres      (Database)
├─ mlflow        (Tracking)
├─ airflow-web   (Orchestration UI)
├─ airflow-sch   (Scheduler)
├─ prometheus    (Metrics)
├─ grafana       (Dashboards)
└─ api           (Inference)
```

**Benefits:**
- ✅ **Reproducibility**: Same env dev→prod
- ✅ **Isolation**: No dependency conflicts
- ✅ **Portability**: Deploy anywhere
- ✅ **Scalability**: Horizontal scaling easy
- ✅ **Rollback**: Version control

**One Command Deploy:**
```bash
docker-compose up -d
# Start entire MLOps stack!
```

**Visual:**
- Docker architecture diagram
- Container orchestration flow

---

## 🎯 SECTION 7: RESULTS & EVALUATION (4 slides)

### Slide 25: Model Comparison
**Content:**
**Experiments Conducted:**

| Model | Accuracy | Precision | Recall | F1-Score | Params | Inference |
|-------|----------|-----------|--------|----------|--------|-----------|
| **EfficientNet B0 (Optimized)** | **95.08%** | **95.21%** | **95.08%** | **95.12%** | 4.0M | **23ms** |
| MobileNetV3 Large | 93.21% | 93.45% | 93.21% | 93.28% | 5.4M | 18ms |
| EfficientNet B0 (Baseline) | 91.45% | 91.67% | 91.45% | 91.51% | 4.0M | 25ms |

**Winner: EfficientNet B0 (Optimized)**
- Best accuracy-speed tradeoff
- Optimizations: mixed precision, LR scheduling
- Production-ready performance

**Visual:**
- Bar chart comparison
- Accuracy vs Inference Time scatter plot

---

### Slide 26: Detailed Performance Metrics
**Content:**
**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bacterial Leaf Blight | 0.96 | 0.95 | 0.95 | 100 |
| Brown Spot | 0.94 | 0.95 | 0.94 | 100 |
| Healthy | 0.98 | 0.97 | 0.97 | 100 |
| Leaf Blast | 0.95 | 0.96 | 0.95 | 100 |
| Leaf Scald | 0.93 | 0.94 | 0.93 | 100 |
| Narrow Brown Spot | 0.95 | 0.94 | 0.94 | 100 |
| **Macro Avg** | **0.952** | **0.951** | **0.951** | **600** |

**Key Insights:**
- ✅ Consistent performance across all classes
- ✅ Healthy class: highest precision (0.98)
- ✅ No significant class bias
- ✅ All classes > 93% F1-score

**Visual:**
- Confusion matrix heatmap
- Per-class F1-score bar chart

---

### Slide 27: System Performance
**Content:**
**API Performance:**
- **Latency**:
  - P50: 20ms
  - P95: 35ms
  - P99: 50ms
- **Throughput**:
  - Single instance: 50 req/s (CPU)
  - With GPU: 200 req/s
- **Uptime**: 99.9%

**Resource Usage:**
- CPU: ~30% (idle), ~80% (load)
- Memory: 2GB (model loaded)
- GPU Memory: 1.5GB (if using GPU)
- Disk: 500MB (model + artifacts)

**Scalability:**
- Horizontal: Load balancer + multiple API containers
- Vertical: GPU acceleration for higher throughput

**Cost Estimate (AWS):**
- EC2 t3.medium: $30/month
- Monitoring: $15/month
- Storage: $5/month
- **Total**: ~$50/month

**Visual:**
- Latency distribution histogram
- Resource usage over time chart

---

### Slide 28: MLOps Metrics
**Content:**
**Development Metrics:**
- Experiments tracked: 10+
- Models registered: 3 versions
- Code coverage: 88%
- Tests passing: 11/11

**Deployment Metrics:**
- Deployment frequency: Weekly (automated)
- Lead time: <30 minutes (code → prod)
- MTTR (Mean Time to Recovery): <5 minutes
- Change failure rate: <5%

**Monitoring Metrics:**
- Metrics collected: 15+
- Dashboard panels: 7
- Alert rules: 5
- Incident response time: <10 minutes

**Compliance:**
- ✅ All tests automated
- ✅ Code reviews required
- ✅ Deployment approval gates
- ✅ Rollback capability

**Visual:**
- MLOps maturity assessment radar chart
- Deployment pipeline timeline

---

## 🎯 SECTION 8: DEMO (2 slides)

### Slide 29: Live Demo Flow
**Content:**
**Demo Checklist:**

1. ✅ **MLflow UI** - Show experiments & model registry
   - Compare runs
   - View artifacts

2. ✅ **API Inference** - Swagger UI
   - Upload test image
   - Get prediction result

3. ✅ **Monitoring** - Grafana dashboard
   - Real-time metrics
   - Request visualization

4. ✅ **Orchestration** - Airflow
   - Training pipeline DAG
   - Deployment pipeline

5. ✅ **Docker** - Container status
   - Show running services
   - Logs inspection

**Demo Time: 5-7 minutes**

**Visual:** Demo flow diagram với screenshots preview

---

### Slide 30: System Demo Screenshots
**Content:**
**Key Interfaces:**

[4-panel layout với screenshots:]

1. **Top-Left**: MLflow UI - Experiments comparison
2. **Top-Right**: Swagger API - Prediction result
3. **Bottom-Left**: Grafana - Metrics dashboard
4. **Bottom-Right**: Airflow - DAG graph view

**Access URLs:**
- MLflow: http://localhost:5000
- API: http://localhost:8000/docs
- Grafana: http://localhost:3000
- Airflow: http://localhost:8080

**Visual:** Grid of 4 screenshots với captions

---

## 🎯 SECTION 9: CONCLUSION (3 slides)

### Slide 31: Key Achievements
**Content:**
**Technical Achievements:**
- ✅ 95.08% accuracy (SOTA for rice diseases)
- ✅ <25ms inference latency
- ✅ End-to-end MLOps pipeline implemented
- ✅ 88% test coverage
- ✅ Automated CI/CD deployment

**MLOps Best Practices:**
- ✅ Experiment tracking (MLflow)
- ✅ Model versioning & registry
- ✅ Automated orchestration (Airflow)
- ✅ Real-time monitoring (Prometheus + Grafana)
- ✅ Containerized deployment (Docker)
- ✅ Comprehensive testing (Pytest)

**Business Impact:**
- Fast & accurate disease detection
- Scalable to 1000s of farmers
- Low operational cost (~$50/month)
- Easy to maintain & update

**Visual:**
- Achievement badges/icons
- Impact metrics dashboard

---

### Slide 32: Lessons Learned & Challenges
**Content:**
**Challenges Faced:**
1. **Data Imbalance** (initial)
   - Solution: Balanced sampling + augmentation

2. **Model Overfitting**
   - Solution: Dropout, weight decay, early stopping

3. **Slow Training**
   - Solution: Mixed precision (AMP), GPU optimization

4. **Deployment Complexity**
   - Solution: Docker Compose simplified orchestration

5. **Monitoring Setup**
   - Solution: Prometheus + Grafana integration

**Key Learnings:**
- MLOps tools giảm thiểu manual work
- Monitoring crucial cho production
- Containerization ensure consistency
- Testing prevent regressions
- Documentation = sustainability

**Visual:**
- Challenge → Solution flow diagram
- Lessons learned icons

---

### Slide 33: Future Work & Improvements
**Content:**
**Short-term (3-6 months):**
- [ ] Mobile app integration (iOS/Android)
- [ ] Multi-language support (Vietnamese, English)
- [ ] Batch prediction API
- [ ] Model ensemble for higher accuracy
- [ ] A/B testing framework

**Long-term (6-12 months):**
- [ ] Disease severity estimation
- [ ] Treatment recommendation system
- [ ] Multi-crop support (rice, wheat, corn)
- [ ] Edge deployment (Raspberry Pi, Jetson)
- [ ] Federated learning for privacy

**Scalability:**
- [ ] Kubernetes deployment
- [ ] Auto-scaling based on load
- [ ] Multi-region deployment
- [ ] CDN for image uploads

**Research:**
- [ ] Attention mechanisms (ViT, SWIN)
- [ ] Few-shot learning (new diseases)
- [ ] Explainability (Grad-CAM)

**Visual:**
- Roadmap timeline
- Feature priority matrix

---

## 🎯 SECTION 10: CLOSING (2 slides)

### Slide 34: References & Resources
**Content:**
**Technologies Used:**
- PyTorch: https://pytorch.org/
- MLflow: https://mlflow.org/
- FastAPI: https://fastapi.tiangolo.com/
- Airflow: https://airflow.apache.org/
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/

**Research Papers:**
- EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan et al., 2019)
- Transfer Learning: "A Survey on Transfer Learning" (Pan & Yang, 2010)

**Dataset:**
- Rice Leaf Disease Image Dataset (Kaggle)

**Code Repository:**
- GitHub: [your-repo-link]
- Documentation: README.md, DEMO.md

**Visual:**
- Logo grid của các technologies
- QR code link to repo

---

### Slide 35: Q&A
**Content:**
```
                  Questions?

        Thank you for your attention!

Contact:
📧 Email: [your-email]
🐱 GitHub: [github-link]
🔗 LinkedIn: [linkedin-link]
```

**Prepared Questions:**
1. How to handle model drift?
2. Why EfficientNet over ResNet?
3. Cost to scale to 10,000 users?
4. How to add new disease classes?
5. Security considerations?

**Visual:**
- Large "Q&A" text
- Contact information
- Project QR code

---

## 📝 BONUS: Backup Slides (optional)

### Backup Slide 1: Detailed Training Curves
**Content:**
- Training/Validation Loss curves
- Training/Validation Accuracy curves
- Learning Rate schedule
- Notes on convergence

**Visual:** Multi-line charts

---

### Backup Slide 2: Confusion Matrix Analysis
**Content:**
- Full confusion matrix (6x6)
- Analysis of misclassifications
- Top confused pairs
- Insights for improvement

**Visual:** Confusion matrix heatmap với annotations

---

### Backup Slide 3: API Error Handling
**Content:**
- HTTP status codes
- Error response format
- Input validation
- Rate limiting
- Security (CORS, authentication)

**Visual:** Code snippets

---

### Backup Slide 4: Cost Analysis Deep Dive
**Content:**
- AWS vs GCP vs Azure comparison
- On-premise vs Cloud
- Cost optimization strategies
- Break-even analysis

**Visual:** Cost comparison table, charts

---

## 🎨 DESIGN GUIDELINES

### Color Scheme
- **Primary**: Blue (#2563EB) - Tech, trust
- **Secondary**: Green (#10B981) - Agriculture, growth
- **Accent**: Orange (#F59E0B) - Alerts, highlights
- **Neutral**: Gray (#6B7280) - Text, backgrounds

### Typography
- **Headers**: Bold, Sans-serif (Montserrat, Inter)
- **Body**: Regular, Sans-serif (Open Sans, Roboto)
- **Code**: Monospace (Fira Code, Courier)

### Visual Elements
- Use icons from: Font Awesome, Material Icons
- Charts: Consistent colors, clear legends
- Screenshots: Add borders, shadows for depth
- Diagrams: Use colors to show flow/categories

### Layout
- **Maximum text per slide**: 6-8 bullet points
- **Font size**: Min 18pt for body, 24pt+ for headers
- **White space**: Don't crowd slides
- **Consistency**: Same layout for similar content types

---

## ⏱️ PRESENTATION TIMING (25 minutes total)

| Section | Slides | Time | Notes |
|---------|--------|------|-------|
| Opening | 1-3 | 2 min | Quick intro |
| Introduction | 4-6 | 3 min | Problem & objectives |
| Dataset | 7-9 | 3 min | Show sample images |
| Architecture | 10-13 | 4 min | Focus on high-level |
| Training | 14-17 | 4 min | Key techniques |
| MLOps | 18-24 | 7 min | **Core focus** |
| Results | 25-28 | 4 min | Metrics & comparison |
| Demo | 29-30 | 5 min | Live demo |
| Conclusion | 31-35 | 3 min | Summary & Q&A |

**Tips:**
- Practice timing for each section
- Have backup slides ready but skip if short on time
- Demo có thể shorten nếu cần
- Reserve 5+ minutes for Q&A

---

## ✅ FINAL CHECKLIST

**Before Presentation:**
- [ ] All services running (docker-compose up)
- [ ] Test images ready for demo
- [ ] Browser tabs pre-opened
- [ ] Backup slides prepared
- [ ] Rehearse timing (20-25 min)
- [ ] Check projector compatibility
- [ ] Have offline version ready

**Slide Quality:**
- [ ] No typos/grammar errors
- [ ] All images high resolution
- [ ] Charts readable from distance
- [ ] Consistent design throughout
- [ ] Slide numbers visible
- [ ] References cited properly

**Demo Preparation:**
- [ ] Test API with sample images
- [ ] Grafana dashboard accessible
- [ ] MLflow experiments visible
- [ ] Airflow DAGs ready to show
- [ ] Backup screenshots if demo fails

---

**Good luck với presentation! 🚀**
