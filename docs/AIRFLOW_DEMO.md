# Demo Airflow - Orchestration (Phần 7)

## 🎯 Mục tiêu Demo
Chứng minh khả năng orchestrate toàn bộ MLOps workflow tự động với Airflow.

## 📋 Chuẩn bị trước khi Demo

### 1. Truy cập Airflow
- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`

### 2. Kiểm tra DAGs có sẵn
Airflow có 2 DAGs chính:
- `rice_disease_training_pipeline` - Training tự động
- `rice_disease_deployment_pipeline` - Deployment tự động

---

## 🎬 Kịch bản Demo (5-7 phút)

### **Bước 1: Giới thiệu giao diện (30s)**

Sau khi login, chỉ vào các phần:
- **DAGs list** - Hiển thị tất cả workflows
- **Status indicators** - Xanh (success), Đỏ (failed), Vàng (running)
- **Schedule** - Tần suất chạy tự động
- **Tags** - Phân loại DAGs

**Nói:** *"Airflow là công cụ orchestration quản lý toàn bộ MLOps workflow. Tôi đã thiết lập 2 pipelines: training và deployment."*

---

### **Bước 2: Demo Training Pipeline (2-3 phút)**

#### 2.1. Mở DAG
1. Click vào **rice_disease_training_pipeline**
2. Bật toggle switch (bên trái DAG name) để enable DAG
3. Click tab **Graph** để xem workflow

#### 2.2. Giải thích workflow
Chỉ vào 5 tasks theo flow:

```
validate_data → setup_dvc → train_model → evaluate_model → notify_completion
```

**Giải thích từng task:**

| Task | Chức năng |
|------|-----------|
| `validate_data` | Kiểm tra data có đủ và hợp lệ không |
| `setup_dvc` | Setup DVC cho data versioning |
| `train_model` | Train model trong Docker container |
| `evaluate_model` | Đánh giá model trên validation set |
| `notify_completion` | Gửi thông báo khi hoàn thành |

**Nói:** *"Pipeline này chạy tự động hàng tuần (weekly). Nếu task nào fail, Airflow sẽ retry 1 lần sau 5 phút."*

#### 2.3. Trigger DAG thủ công
1. Click nút ▶️ **Trigger DAG** (góc trên bên phải)
2. Click **Trigger** trong popup
3. DAG run mới xuất hiện với màu vàng (running)

**Nói:** *"Tôi đang trigger pipeline thủ công để demo. Trong production, nó sẽ chạy tự động theo schedule."*

#### 2.4. Xem execution details
1. Click vào DAG run (hàng mới màu vàng/xanh)
2. Click vào task **validate_data** → **Log**
3. Chỉ vào output: "✓ Training samples: X", "✓ Validation samples: Y"

**Nói:** *"Mỗi task có logs chi tiết. Task validate_data kiểm tra xem có đủ dữ liệu hay không."*

---

### **Bước 3: Demo Deployment Pipeline (2 phút)**

#### 3.1. Mở DAG
1. Quay lại **DAGs** (click logo Airflow)
2. Click vào **rice_disease_deployment_pipeline**
3. Click tab **Graph**

#### 3.2. Giải thích workflow

```
validate_model → build_docker_image → deploy_to_staging
                                            ↓
                                    run_smoke_tests → deploy_to_production
```

**Giải thích:**

| Task | Chức năng |
|------|-----------|
| `validate_model` | Kiểm tra model accuracy > 80% |
| `build_docker_image` | Build Docker image cho API |
| `deploy_to_staging` | Deploy lên staging environment |
| `run_smoke_tests` | Test API health endpoint |
| `deploy_to_production` | Deploy production nếu test pass |

**Nói:** *"Deployment pipeline được trigger khi có model mới tốt hơn. Nó tự động build Docker image, deploy staging, chạy smoke tests, rồi mới deploy production."*

#### 3.3. Highlight tính năng CI/CD
**Chỉ vào các điểm:**
- **Automatic validation** - Model phải pass quality threshold
- **Staging first** - Test trước khi deploy production
- **Smoke tests** - Đảm bảo API hoạt động đúng
- **Rollback capability** - Có thể rollback nếu có vấn đề

---

### **Bước 4: Demo Monitoring & Alerting (1 phút)**

#### 4.1. Xem DAG Runs History
1. Quay lại DAGs list
2. Click vào số trong cột **Runs** (ví dụ: 3 success, 1 failed)
3. Hiển thị tất cả lần chạy với timestamp

**Nói:** *"Airflow lưu lại lịch sử tất cả runs. Nếu có fail, chúng ta có thể xem logs để debug."*

#### 4.2. Xem Task Duration
1. Click vào 1 successful DAG run
2. Click tab **Gantt** (hoặc **Duration**)
3. Chỉ vào biểu đồ thời gian của từng task

**Nói:** *"Biểu đồ Gantt cho thấy task nào tốn thời gian nhất. Giúp optimize pipeline."*

#### 4.3. Alert Configuration
Click vào **Admin** → **Connections** (nếu có thời gian)

**Nói:** *"Airflow có thể gửi alert qua email, Slack khi có task fail. Trong production, chúng ta config alerts để monitor 24/7."*

---

### **Bước 5: Tổng kết (30s)**

**Điểm nhấn:**
- ✅ **Automation** - Training tự động hàng tuần, không cần can thiệp thủ công
- ✅ **Reliability** - Retry mechanism, error handling
- ✅ **Visibility** - Logs chi tiết, visualization workflow
- ✅ **Scalability** - Dễ dàng thêm tasks mới vào pipeline
- ✅ **CI/CD Integration** - Deployment tự động với validation gates

**Nói:** *"Airflow giúp chúng ta orchestrate toàn bộ MLOps lifecycle - từ data validation, training, evaluation đến deployment production. Mọi thứ tự động, có monitoring và có thể rollback khi cần."*

---

## 🎯 Q&A Thường gặp

### Q1: "Airflow khác gì với cron job?"
**A:** Airflow có:
- Dependency management (task A phải chạy xong mới chạy task B)
- Retry mechanism tự động
- Web UI để monitor
- Centralized logging
- Dynamic pipeline generation

### Q2: "Làm sao để schedule training hàng ngày thay vì hàng tuần?"
**A:** Sửa `schedule_interval` trong DAG:
```python
schedule_interval="@daily"  # Hoặc "0 0 * * *" cho midnight
```

### Q3: "Airflow chạy tasks ở đâu?"
**A:**
- Tasks đơn giản (PythonOperator) chạy trong Airflow worker
- Tasks nặng (train model) chạy trong Docker container riêng biệt
- Có thể scale với Kubernetes executor

### Q4: "Làm sao để trigger deployment pipeline từ training pipeline?"
**A:** Thêm TriggerDagRunOperator vào cuối training pipeline:
```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_deployment = TriggerDagRunOperator(
    task_id='trigger_deployment',
    trigger_dag_id='rice_disease_deployment_pipeline',
    dag=dag
)
```

---

## 📊 Metrics để Demo (Nếu có thời gian)

### Xem Task Success Rate
1. Click **Browse** → **Task Instances**
2. Filter by DAG
3. Chỉ vào success/failed ratio

### Xem Execution Time Trends
1. Click vào DAG
2. Tab **Landing Times** hoặc **Task Duration**
3. Chỉ vào trend line

---

## 🚀 Tips cho Demo mượt mà

1. **Chuẩn bị trước:**
   - Enable cả 2 DAGs trước khi demo
   - Trigger 1 lần để có history
   - Bookmark các tabs: DAGs list, Training pipeline, Deployment pipeline

2. **Trong lúc demo:**
   - Giữ ngắn gọn, tập trung vào workflow
   - Không đợi task chạy xong (quá lâu)
   - Nếu task fail, dùng làm case study để giải thích retry mechanism

3. **Backup plan:**
   - Nếu Airflow không load được, chỉ vào code DAG (training_pipeline.py)
   - Giải thích workflow qua code thay vì UI

---

## ✅ Checklist trước khi Demo

- [ ] Airflow webserver đang chạy (http://localhost:8080)
- [ ] Login với admin/admin thành công
- [ ] Cả 2 DAGs hiển thị trong list
- [ ] Enable toggle cho cả 2 DAGs
- [ ] Có ít nhất 1 successful run (trigger trước để có history)
- [ ] Đã đọc qua logs của các tasks

---

**Thời gian demo**: 5-7 phút
**Độ khó**: Trung bình
**Impact**: Cao - Chứng minh automation & orchestration capability
