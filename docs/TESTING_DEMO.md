# Demo Testing & Quality Assurance (Phần 9)

## 🎯 Mục tiêu Demo
Chứng minh code quality cao với comprehensive test suite, đảm bảo reliability và maintainability.

## 📋 Chuẩn bị

### Cài đặt test dependencies (nếu chưa có)
```powershell
pip install pytest pytest-cov pytest-xdist httpx
```

### Kiểm tra test files có sẵn
```powershell
ls tests/
```

**Test files:**
- `test_data.py` - Data pipeline tests
- `test_model.py` - Model architecture tests
- `test_api.py` - API integration tests
- `conftest.py` - Test fixtures và configuration

---

## 🎬 Kịch bản Demo (7-10 phút)

### **Bước 1: Giới thiệu Test Strategy (1 phút)**

**Hiển thị test structure:**
```powershell
tree tests /F
```

**Giải thích Test Pyramid:**
```
        /\
       /  \     E2E Tests (ít)
      /----\
     /      \   Integration Tests (vừa)
    /--------\
   /          \ Unit Tests (nhiều)
  /------------\
```

**Nói:** *"Chúng ta follow test pyramid: nhiều unit tests, vừa phải integration tests. Đảm bảo code quality và prevent regressions."*

---

### **Bước 2: Demo Unit Tests - Data Module (2 phút)**

#### 2.1. Xem test code
```powershell
code tests/test_data.py
```

**Highlight test cases:**
- ✅ `test_rice_dataset` - Dataset creation
- ✅ `test_dataset_getitem` - Data loading
- ✅ `test_train_transforms` - Augmentation
- ✅ `test_class_distribution` - Data balance

#### 2.2. Run data tests
```powershell
pytest tests/test_data.py -v
```

**Expected output:**
```
tests/test_data.py::test_rice_dataset PASSED
tests/test_data.py::test_dataset_getitem PASSED
tests/test_data.py::test_train_transforms PASSED
tests/test_data.py::test_val_transforms PASSED
tests/test_data.py::test_class_distribution PASSED

====== 5 passed in 2.34s ======
```

**Nói:** *"Data tests đảm bảo data pipeline hoạt động đúng - load data, transforms, class distribution."*

---

### **Bước 3: Demo Unit Tests - Model Module (2 phút)**

#### 3.1. Xem test code
```powershell
code tests/test_model.py
```

**Highlight:**
```python
def test_model_forward():
    """Test model forward pass."""
    model = create_model("efficientnet_b0", num_classes=6)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert output.shape == (4, 6)
```

**Parametrized tests:**
```python
@pytest.mark.parametrize("num_classes", [2, 6, 10])
def test_different_num_classes(num_classes):
    # Test với multiple configurations
```

#### 3.2. Run model tests
```powershell
pytest tests/test_model.py -v
```

**Expected output:**
```
tests/test_model.py::test_create_model PASSED
tests/test_model.py::test_model_forward PASSED
tests/test_model.py::test_model_parameters PASSED
tests/test_model.py::test_different_num_classes[2] PASSED
tests/test_model.py::test_different_num_classes[6] PASSED
tests/test_model.py::test_different_num_classes[10] PASSED

====== 6 passed in 3.21s ======
```

**Nói:** *"Model tests verify architecture đúng - output shapes, parameter counts, compatibility với different num_classes."*

---

### **Bước 4: Demo Integration Tests - API (2 phút)**

#### 4.1. Xem test code
```powershell
code tests/test_api.py
```

**Highlight fixtures:**
```python
@pytest.fixture
def client():
    from api.app import app
    return TestClient(app)

@pytest.fixture
def sample_image():
    img = Image.new("RGB", (224, 224))
    # ... return image bytes
```

**Test cases:**
- ✅ `test_root_endpoint` - Root route
- ✅ `test_health_endpoint` - Health check
- ✅ `test_predict_endpoint_with_image` - Inference

#### 4.2. Run API tests
```powershell
pytest tests/test_api.py -v
```

**Expected output:**
```
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_predict_endpoint_no_file PASSED
tests/test_api.py::test_predict_endpoint_with_image PASSED

====== 5 passed in 1.87s ======
```

**Nói:** *"API tests ensure endpoints hoạt động đúng - status codes, response format, error handling."*

---

### **Bước 5: Demo Test Coverage (1 phút)**

#### 5.1. Run tests với coverage
```powershell
pytest --cov=src --cov=api --cov-report=html --cov-report=term
```

**Expected output:**
```
---------- coverage: platform win32, python 3.9 ----------
Name                    Stmts   Miss  Cover
-------------------------------------------
api\__init__.py             1      0   100%
api\app.py                 89     12    87%
src\__init__.py             2      0   100%
src\data\__init__.py       15      2    87%
src\data\dataset.py        45      5    89%
src\models\__init__.py     12      1    92%
src\models\classifier.py   67      8    88%
-------------------------------------------
TOTAL                     231     28    88%
```

**Nói:** *"88% coverage - majority của code được test. Mục tiêu > 80% coverage."*

#### 5.2. Xem HTML coverage report
```powershell
start htmlcov/index.html
```

**Chỉ vào:**
- Green lines - covered
- Red lines - not covered
- Identify untested code paths

---

### **Bước 6: Demo Parallel Testing (1 phút)**

#### 6.1. Run tests in parallel
```powershell
pytest -n auto -v
```

**Output:**
```
gw0 [16] / gw1 [16] / gw2 [16] / gw3 [16]
... tests running in parallel ...

====== 16 passed in 1.2s (0:00:01) ======
```

**Nói:** *"Pytest-xdist chạy tests parallel. Từ 5 seconds xuống còn 1.2 seconds - save time trong CI/CD."*

---

### **Bước 7: Demo Test in Docker (1 phút)**

#### 7.1. Create test Dockerfile (giải thích)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["pytest", "-v", "--cov=src", "--cov=api"]
```

#### 7.2. Run tests in container
```powershell
docker build -t rice-disease-tests -f docker/Dockerfile.test .
docker run --rm rice-disease-tests
```

**Nói:** *"Tests chạy trong Docker để ensure consistency. Same environment cho CI/CD."*

---

### **Bước 8: Demo CI/CD Integration (1 phút)**

#### 8.1. GitHub Actions workflow (giải thích)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

**Nói:** *"Tests tự động chạy mỗi khi push code. Block merge nếu tests fail hoặc coverage drop."*

---

### **Bước 9: Demo Model Quality Tests (1 phút)**

#### 9.1. Model performance tests
```powershell
code tests/test_model_quality.py
```

**Test minimum accuracy:**
```python
def test_model_accuracy():
    """Test model meets minimum accuracy threshold."""
    model = load_model("models/best_model.pth")
    val_loader = create_dataloader("validation/", batch_size=32)

    accuracy = evaluate(model, val_loader)
    assert accuracy >= 0.85, f"Accuracy {accuracy} below threshold 0.85"

def test_inference_speed():
    """Test inference speed within acceptable range."""
    model = load_model("models/best_model.pth")
    img = torch.randn(1, 3, 224, 224)

    import time
    start = time.time()
    output = model(img)
    duration = time.time() - start

    assert duration < 0.5, f"Inference took {duration}s, threshold 0.5s"
```

**Nói:** *"Quality tests ensure model performance - minimum accuracy, inference speed, no degradation over time."*

---

### **Bước 10: Demo Stress Testing (optional)**

#### 10.1. API load testing
```powershell
# Install locust
pip install locust

# Create locustfile.py
code tests/locustfile.py
```

```python
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        files = {'file': open('test_image.jpg', 'rb')}
        self.client.post("/predict", files=files)
```

**Run load test:**
```powershell
locust -f tests/locustfile.py --host=http://localhost:8000
```

**Nói:** *"Load testing đảm bảo API handle concurrent requests. Monitor response time, error rate."*

---

### **Bước 11: Tổng kết (30s)**

**Test Coverage Summary:**
- ✅ **Unit Tests** - 11 tests cho data, model modules (88% coverage)
- ✅ **Integration Tests** - 5 tests cho API endpoints
- ✅ **Quality Tests** - Model accuracy, inference speed
- ✅ **CI/CD** - Automated testing trong pipeline
- ✅ **Parallel Execution** - Fast feedback (1.2s total)

**Quality Metrics:**
- 📊 **Test Coverage**: 88% (target > 80%)
- ⚡ **Test Speed**: 1.2s parallel, 5s sequential
- ✅ **Pass Rate**: 16/16 tests passing
- 🔒 **No Regressions**: Tests prevent code quality degradation

**Nói:** *"Comprehensive test suite đảm bảo code quality. Tests chạy tự động, catch bugs early, enable safe refactoring."*

---

## 🎯 Q&A Thường gặp

### Q1: "80% coverage có đủ không?"
**A:**
- 80% là baseline tốt
- Critical paths (inference, data loading) phải 100%
- Config/utils có thể lower coverage
- Focus on meaningful tests, không phải chỉ coverage number

### Q2: "Unit test vs Integration test - khi nào dùng gì?"
**A:**
- **Unit tests** - Test 1 function/class isolated, fast, nhiều
- **Integration tests** - Test components work together, slower, ít hơn
- **E2E tests** - Test full workflow, slowest, rất ít

### Q3: "Làm sao để test model training?"
**A:**
```python
def test_training_step():
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 1])

    # Forward pass
    output = model(x)
    loss = F.cross_entropy(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients exist
    assert all(p.grad is not None for p in model.parameters())
```

### Q4: "Test trong Docker vs local có khác biệt?"
**A:**
- Docker: Consistent environment, same as CI/CD
- Local: Faster, easier to debug
- Best practice: Local cho dev, Docker cho CI/CD

### Q5: "Làm sao để test async code trong FastAPI?"
**A:**
```python
import pytest

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/async-endpoint")
    assert response.status_code == 200
```

---

## 📊 Demo Metrics

### Test Execution Time
```powershell
pytest --durations=10
```

### Test Coverage by Module
```powershell
pytest --cov=src --cov-report=term-missing
```

### Failed Tests Details
```powershell
pytest -v --tb=short
```

---

## 🚀 Tips cho Demo mượt mà

1. **Chuẩn bị trước:**
   - Run tests 1 lần để ensure all passing
   - Generate coverage report HTML
   - Có backup screenshots nếu tests fail

2. **Trong lúc demo:**
   - Start với simple unit tests
   - Show test code → Run tests → Show results
   - Highlight coverage gaps → Explain strategy

3. **Visual aids:**
   - Coverage report HTML (green/red highlighting)
   - pytest output với colors (-v flag)
   - CI/CD pipeline screenshot (GitHub Actions)

4. **Common issues:**
   - Import errors → Check PYTHONPATH
   - Model not found → Skip model-dependent tests với `pytest.skip()`
   - Slow tests → Use `-k` to run subset

---

## ✅ Checklist trước khi Demo

- [ ] All tests passing: `pytest`
- [ ] Coverage report generated: `pytest --cov --cov-report=html`
- [ ] No warnings: `pytest -W error::DeprecationWarning`
- [ ] Tests run in Docker: `docker build -t tests .`
- [ ] Có test code mở sẵn trong editor
- [ ] Terminal clean (clear screen)

---

## 🔗 Quick Demo Commands

```powershell
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=src --cov=api --cov-report=html

# Run parallel
pytest -n auto

# Run and stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf

# Show print statements
pytest -s

# Run tests matching pattern
pytest -k "test_model"

# Generate JUnit XML report (for CI/CD)
pytest --junitxml=test-results.xml
```

---

## 📝 Bonus: Create Missing Tests

### Test model quality
```python
# tests/test_model_quality.py
def test_model_accuracy_threshold():
    """Ensure model meets minimum accuracy."""
    # Load validation data
    # Run inference
    # Assert accuracy >= 85%
    pass

def test_no_overfitting():
    """Check train vs val accuracy gap."""
    # Load metrics
    # Assert (train_acc - val_acc) < 10%
    pass
```

### Test data validation
```python
# tests/test_data_validation.py
def test_no_corrupted_images():
    """Check all images can be loaded."""
    # Iterate through dataset
    # Try to load each image
    # Assert no errors
    pass

def test_class_balance():
    """Check classes are reasonably balanced."""
    # Get class distribution
    # Assert max_class / min_class < 3
    pass
```

---

**Thời gian demo**: 7-10 phút
**Độ khó**: Trung bình-Cao
**Impact**: Rất cao - Chứng minh code quality và reliability
