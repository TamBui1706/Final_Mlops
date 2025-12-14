# 🏆 Hướng Dẫn Chi Tiết: Model Registry & Versioning Demo

## 📋 Tổng Quan
Model Registry là nơi quản lý, version control và track các models đã trained. MLflow Registry cho phép:
- ✅ Lưu trữ nhiều versions của model
- ✅ Gắn tags và stage (None → Staging → Production)
- ✅ Link back to training run
- ✅ Rollback dễ dàng nếu cần

---

## 🎯 Bước 1: Kiểm Tra Models Đã Register

### Mở MLflow UI
```powershell
# Nếu chưa start MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

Mở browser: **http://localhost:5000**

### Check Models Tab
1. Click tab **"Models"** ở top menu
2. Xem danh sách registered models

**Nếu CHƯA có model nào:**
- List sẽ rỗng
- Cần register model mới (xem Bước 2)

**Nếu ĐÃ có models:**
- Sẽ thấy model name: `rice-disease-classifier`
- Click vào để xem details

---

## 🎯 Bước 2: Register Model Từ Best Run

### Option A: Register qua MLflow UI (Khuyến nghị cho demo)

**Step 1: Tìm Best Run**
1. Click tab **"Experiments"**
2. Click experiment **"Rice Disease Classification"**
3. Sort by `val_accuracy` (descending)
4. Best run: **EfficientNet-B0 Optimized - 98.67%**
   - Run ID: `4b8e6057500b4b03bef452bac0c212dd`
   - Accuracy: 0.9867

**Step 2: Register Model**
1. Click vào best run đó
2. Scroll xuống **"Artifacts"** section
3. **LƯU Ý:** Nếu không thấy folder `model/` trong artifacts:
   - Model chưa được log as MLflow model
   - Cần log model trước (xem Option B)

4. **Nếu có folder `model/`:**
   - Click vào folder `model/`
   - Click button **"Register Model"** (top right)
   - Điền thông tin:
     - **Model Name:** `rice-disease-classifier`
     - **Description:** "Production model for rice leaf disease classification - EfficientNet-B0 Optimized"
   - Click **"Register"**

### Option B: Register bằng Python Script

**Nếu model chưa được log to MLflow, cần log lại:**

```powershell
# Chạy script để log và register model
python register_model.py
```

**Script này sẽ:**
1. Load model checkpoint từ `models/best_model.pth`
2. Log model lên MLflow run
3. Register model to Model Registry
4. Set stage = Production

**Hoặc tự viết script ngắn gọn:**

```python
import mlflow
import torch
from src.models.efficientnet import EfficientNetB0

mlflow.set_tracking_uri("http://localhost:5000")

# Best run ID
RUN_ID = "4b8e6057500b4b03bef452bac0c212dd"

# Load model
model = EfficientNetB0(num_classes=6)
checkpoint = torch.load("models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Log model to run
with mlflow.start_run(run_id=RUN_ID):
    mlflow.pytorch.log_model(model, "model")
    print("✅ Model logged to MLflow")

# Register model
model_uri = f"runs:/{RUN_ID}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name="rice-disease-classifier"
)

print(f"✅ Model registered: {result.name}, version {result.version}")
```

---

## 🎯 Bước 3: Demo Model Registry trong MLflow UI

### 3.1 Navigate to Model Registry

1. **Click tab "Models"** ở top menu
2. Bạn sẽ thấy model: **`rice-disease-classifier`**

**Giải thích cho audience:**
> "Đây là Model Registry - nơi quản lý tất cả versions của models. Mỗi model có thể có nhiều versions, mỗi version link back to training run cụ thể."

### 3.2 Click vào Model Name

Sẽ thấy trang chi tiết với các sections:

#### **Latest Versions Section**
Shows các versions mới nhất theo stage:

| Version | Stage | Registered | Run ID | Metrics |
|---------|-------|------------|--------|---------|
| Version 1 | Production | 2025-12-14 | 4b8e60... | val_acc: 0.9867 |
| Version 2 | Staging | 2025-12-14 | 6b841f... | val_acc: 0.9621 |
| Version 3 | None | 2025-12-14 | adf0b8... | val_acc: 0.9489 |

**Giải thích:**
- **Production:** Model đang chạy trên production
- **Staging:** Model đang test trên staging environment
- **None:** Model mới register, chưa deploy

### 3.3 Click vào Version (VD: Version 1)

Sẽ thấy trang **Model Version Details:**

#### **Overview Tab:**
- **Version:** 1
- **Stage:** Production
- **Created:** 2025-12-14 10:30:45
- **Created By:** user
- **Source Run:** [Link to run 4b8e60...]
- **Model URI:** `runs:/4b8e6057500b4b03bef452bac0c212dd/model`
- **Description:** Production model for rice leaf disease...

#### **Schema Tab:**
- Input schema (image tensors)
- Output schema (predictions)

#### **Source Run Link:**
Click vào **"Source Run"** → Quay lại experiment run
- Xem lại metrics, parameters, artifacts
- Full traceability!

**Giải thích cho audience:**
> "Mỗi model version được link chặt chẽ với training run. Tôi có thể click vào Source Run để xem lại toàn bộ metrics, parameters, confusion matrix của lần training đó. Full traceability!"

---

## 🎯 Bước 4: Demo Version Management

### 4.1 Transition Model Stage

**Demo scenario:** Promote model từ Staging → Production

**Trong Model Version page:**
1. Click button **"Stage"** (top right)
2. Select **"Transition to → Production"**
3. Dialog xuất hiện:
   - **Archive existing Production versions:** ✅ (check this)
   - **Description:** "Promoting v2 to production - higher accuracy"
4. Click **"OK"**

**Kết quả:**
- Version 2 → Production
- Version 1 → Archived (hoặc None)

**Giải thích:**
> "Khi có model mới tốt hơn, tôi chỉ cần click vài cái để promote lên Production. Model cũ được archive tự động. API sẽ tự động load version mới."

### 4.2 Demo Rollback

**Demo scenario:** Rollback về version cũ nếu version mới có vấn đề

1. Vào **Version 1** (version cũ)
2. Click **"Stage" → "Transition to → Production"**
3. Confirm

**Kết quả:**
- Version 1 quay lại Production
- Version 2 → Archived/None

**Giải thích:**
> "Nếu version mới có vấn đề, rollback rất đơn giản - chỉ cần promote version cũ lại. Zero downtime!"

---

## 🎯 Bước 5: Demo Model Metadata & Tags

### 5.1 Add Tags to Model

**Trong Model page (rice-disease-classifier):**

1. Scroll xuống section **"Tags"**
2. Click **"Add Tag"**
3. Thêm các tags:
   - `task`: `image-classification`
   - `framework`: `pytorch`
   - `architecture`: `efficientnet-b0`
   - `dataset`: `rice-leaf-disease`
   - `accuracy`: `98.67%`
   - `production-ready`: `true`

**Giải thích:**
> "Tags giúp tìm kiếm và filter models dễ dàng. Rất hữu ích khi có nhiều models."

### 5.2 Update Description

1. Click **"Edit"** ở Description section
2. Cập nhật:
```
Production model for Rice Leaf Disease Classification

**Architecture:** EfficientNet-B0 Optimized
**Accuracy:** 98.67% on validation set (528 images)
**Inference Time:** ~23ms
**Model Size:** 20MB
**Classes:** 6 (bacterial_leaf_blight, brown_spot, healthy, leaf_blast, leaf_scald, narrow_brown_spot)

**Training Details:**
- Dataset: 2,100 train + 528 val images
- Augmentation: Albumentations (10+ techniques)
- Optimizer: AdamW with Cosine Annealing
- Mixed Precision: FP16
- Early Stopping: Patience 10 epochs

**Deployment:**
- Stage: Production
- API Endpoint: http://localhost:8000/predict
- Docker Image: rice-disease-api:latest
```

3. Click **"Save"**

---

## 🎯 Bước 6: Compare Model Versions

### 6.1 Compare Feature

**Trong Models page:**
1. Select multiple versions (checkbox)
2. Click **"Compare"** button

**Compare view shows:**
- Side-by-side metrics comparison
- Parameters diff
- Training time
- Model size

**Ví dụ comparison:**

| Metric | Version 1 (Optimized) | Version 2 (Baseline) | Version 3 (MobileNet) |
|--------|----------------------|---------------------|---------------------|
| val_accuracy | **0.9867** 🏆 | 0.9621 | 0.9489 |
| val_f1_score | **0.9863** | 0.9615 | 0.9482 |
| training_time | 15.2 min | 12.8 min | **10.5 min** |
| model_size | 20 MB | 20 MB | **15 MB** |
| inference_time | 23 ms | 24 ms | **18 ms** |

**Giải thích:**
> "Compare feature giúp quyết định model nào tốt hơn. Version 1 có accuracy cao nhất, phù hợp cho production mặc dù inference time hơi chậm hơn MobileNet."

---

## 🎯 Bước 7: Integrate với API

### 7.1 API Load Model From Registry

**Show code trong `api/app.py`:**

```python
import mlflow

# Load model from MLflow Registry
model_name = "rice-disease-classifier"
stage = "Production"  # or "Staging"

model_uri = f"models:/{model_name}/{stage}"
model = mlflow.pytorch.load_model(model_uri)

print(f"✅ Loaded model: {model_name} ({stage})")
```

**Giải thích:**
> "API tự động load model từ Registry với stage Production. Khi promote version mới, API sẽ tự động load version đó sau khi restart. Không cần manually copy model files!"

### 7.2 Test API với Different Versions

**Terminal:**
```powershell
# Test với Production model
curl http://localhost:8000/model/info

# Response:
# {
#   "model_name": "rice-disease-classifier",
#   "version": "1",
#   "stage": "Production",
#   "accuracy": 0.9867,
#   "classes": ["bacterial_leaf_blight", "brown_spot", ...]
# }
```

---

## 🎯 Bước 8: Demo Scenarios

### Scenario 1: New Model Training

**Story:** Data scientist train model mới tốt hơn

**Steps:**
1. Train new model → MLflow logs metrics
2. Compare với Production model trong MLflow
3. Nếu tốt hơn → Register as new version
4. Transition to Staging
5. Test trên staging environment
6. Nếu OK → Transition to Production
7. API automatically loads new version

### Scenario 2: Model Has Issues

**Story:** Production model có accuracy drop

**Steps:**
1. Monitor dashboard phát hiện vấn đề
2. Vào MLflow Registry
3. Rollback to previous version (1 click)
4. Restart API
5. System back to normal

### Scenario 3: A/B Testing

**Story:** Test 2 models simultaneously

**Steps:**
1. Version 1 → Production (80% traffic)
2. Version 2 → Staging (20% traffic)
3. Compare metrics
4. Winner → Production (100%)

---

## 📊 Demo Talking Points

### Key Messages:

1. **Version Control cho ML Models**
   > "Giống như Git cho code, MLflow Registry là Git cho models. Mọi thay đổi đều được track."

2. **Traceability**
   > "Từ model version, tôi có thể trace back đến exact training run, xem lại toàn bộ configs, data, metrics. Reproducibility 100%!"

3. **Easy Rollback**
   > "Production model có vấn đề? Rollback trong 1 phút. Zero stress!"

4. **Collaboration**
   > "Team có thể share models dễ dàng. Data scientist train xong, DevOps deploy luôn. Không cần manually copy files."

5. **Compliance & Governance**
   > "Mọi model changes đều được log. Ai promote, khi nào, tại sao. Critical cho industries regulated."

---

## ❓ Q&A Prep

**Q: MLflow Registry khác gì so với chỉ save model files?**
> A: Registry cung cấp version control, stage management, metadata, tags, và link back to training runs. File system chỉ là folder with .pth files.

**Q: Làm sao API biết load model version nào?**
> A: API config load model với stage "Production". MLflow tự động resolve to latest Production version.

**Q: Có thể deploy nhiều versions simultaneously không?**
> A: Có. Deploy version 1 trên production (port 8000), version 2 trên staging (port 8001). Canary deployment hoặc A/B testing.

**Q: Model Registry có scale với nhiều models không?**
> A: Có. MLflow Registry support unlimited models và versions. Large companies có hàng ngàn models.

**Q: Registry có backup/restore không?**
> A: MLflow Registry sử dụng database backend (PostgreSQL). Database được backup theo schedule.

---

## 🎬 Demo Script Mẫu

**[OPEN MLflow UI - Models Tab]**

> "Bây giờ chúng ta sang Model Registry. Đây là nơi quản lý tất cả models đã trained."

**[CLICK vào rice-disease-classifier]**

> "Model rice-disease-classifier của chúng ta có 3 versions. Version 1 đang ở stage Production với accuracy 98.67% - đây là model đang serve API."

**[CLICK vào Version 1]**

> "Mỗi version có full metadata: khi nào tạo, ai tạo, metrics gì. Quan trọng nhất là link back to Source Run."

**[CLICK Source Run]**

> "Click vào đây, tôi quay lại exact training run. Full traceability - tôi biết model này được train với configs gì, data nào, metrics như thế nào."

**[BACK to Model Registry]**

> "Giả sử tôi có model mới tốt hơn. Tôi chỉ cần..."

**[CLICK Stage → Transition to Production]**

> "...promote model mới lên Production. Version cũ tự động archived. API sẽ load version mới sau khi restart."

**[Show API response]**

> "Và nếu có vấn đề, rollback cũng dễ dàng - chỉ là 1 click. Đây là lý do Model Registry rất quan trọng trong MLOps."

---

## ✅ Checklist Demo Model Registry

- [ ] MLflow UI đang chạy: http://localhost:5000
- [ ] Có ít nhất 1 model đã register
- [ ] Model có ít nhất 2-3 versions
- [ ] 1 version ở stage Production
- [ ] Model versions có descriptions đầy đủ
- [ ] Tags đã được thêm vào model
- [ ] API đang load model từ Registry
- [ ] Biết run IDs của các training runs
- [ ] Chuẩn bị sẵn script để register model (backup)

---

**Phần này rất quan trọng vì nó thể hiện production-readiness của system!** 🏆
