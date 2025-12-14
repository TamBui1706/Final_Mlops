# 🚀 Hướng Dẫn CI/CD Đơn Giản

## 📋 Tổng Quan

Project này có **3 workflows CI/CD tự động**:

| Workflow | Khi nào chạy | Làm gì |
|----------|--------------|--------|
| **CI** | Mỗi khi push code | Test code, kiểm tra lỗi |
| **Deploy** | Push lên main | Build Docker, deploy app |
| **Train Model** | Mỗi Chủ nhật / Manual | Train lại model tự động |

---

## 🔧 1. CI Pipeline - Kiểm tra Code

**File:** `.github/workflows/ci.yml`

### Chức năng:
- ✅ Chạy unit tests
- ✅ Kiểm tra code quality (Black, Flake8)
- ✅ Build Docker image test

### Khi nào chạy:
```bash
# Tự động chạy khi:
git push origin main
git push origin develop

# Hoặc khi tạo Pull Request
```

### Xem kết quả:
1. Vào GitHub repository
2. Click tab **Actions**
3. Chọn workflow **"CI - Test và Kiểm tra Code"**
4. Xem logs chi tiết

---

## 🚀 2. Deploy Pipeline - Deploy Application

**File:** `.github/workflows/deploy.yml`

### Chức năng:
- 🔨 Build Docker images (API + Training)
- 🧪 Test images
- 🚀 Deploy lên server

### Khi nào chạy:
```bash
# Tự động khi push lên main:
git push origin main

# Hoặc manual trigger trên GitHub Actions UI

# Hoặc khi tạo tag version:
git tag v1.0.0
git push origin v1.0.0
```

### Cấu hình deployment:

**Bước 1:** Thêm secrets vào GitHub
```
Settings → Secrets → Actions → New repository secret
```

Cần thêm:
- `DEPLOY_HOST`: địa chỉ server
- `DEPLOY_USER`: username SSH
- `SSH_PRIVATE_KEY`: SSH key để connect

**Bước 2:** Uncomment dòng deploy trong file workflow:
```yaml
# Tìm dòng này và bỏ dấu #
# docker-compose up -d api mlflow
```

---

## 🤖 3. Train Model Pipeline - Training Tự động

**File:** `.github/workflows/train-model.yml`

### Chức năng:
- 📊 Kiểm tra data
- 🚂 Train model tự động
- 💾 Lưu model artifacts

### Khi nào chạy:

**Tự động:** Mỗi Chủ nhật lúc 2 giờ sáng

**Manual:**
1. Vào GitHub → Actions
2. Chọn **"Train Model - Tự động Training"**
3. Click **"Run workflow"**
4. Nhập số epochs (default: 50)
5. Click **"Run workflow"**

### Lấy trained model:
1. Vào workflow run đã hoàn thành
2. Scroll xuống **Artifacts**
3. Download **"trained-model"**

---

## 📊 4. Xem Kết Quả CI/CD

### Trên GitHub:
```
Repository → Actions tab → Chọn workflow
```

### Status badges (thêm vào README):
```markdown
![CI Status](https://github.com/USERNAME/REPO/actions/workflows/ci.yml/badge.svg)
![Deploy Status](https://github.com/USERNAME/REPO/actions/workflows/deploy.yml/badge.svg)
```

---

## 🔍 5. Debug Khi Pipeline Fail

### Bước 1: Xem logs
```
Actions → Click vào run bị đỏ → Click vào job bị lỗi → Xem logs
```

### Bước 2: Test locally
```bash
# Test giống như CI
pip install pytest flake8 black
pytest tests/ -v
black --check src/
flake8 src/

# Build Docker
docker build -f docker/Dockerfile.api -t rice-api:test .
```

### Lỗi thường gặp:

**1. Tests fail:**
```bash
# Fix: Chạy tests locally và fix lỗi
pytest tests/ -v
```

**2. Docker build fail:**
```bash
# Fix: Build locally và xem lỗi
docker build -f docker/Dockerfile.api .
```

**3. Permission denied:**
```
# Fix: Thêm permissions vào workflow file
permissions:
  contents: read
  packages: write
```

---

## 🎯 6. Workflow Thực Tế

### Scenario 1: Thêm feature mới

```bash
# 1. Tạo branch mới
git checkout -b feat/new-feature

# 2. Code feature
# ... viết code ...

# 3. Test locally
pytest tests/
black src/

# 4. Commit và push
git add .
git commit -m "feat: thêm feature mới"
git push origin feat/new-feature

# 5. Tạo Pull Request
# → CI tự động chạy test

# 6. Merge vào main
# → Deploy tự động chạy
```

### Scenario 2: Train model mới

```bash
# Option 1: Manual trigger
# Vào GitHub Actions → Train Model → Run workflow

# Option 2: Đợi schedule
# Tự động chạy mỗi Chủ nhật

# Option 3: Train locally
python src/train.py --epochs 50
python register_model.py
```

### Scenario 3: Deploy lên production

```bash
# 1. Đảm bảo code đã test
git checkout main
git pull

# 2. Tag version mới
git tag v1.2.0
git push origin v1.2.0

# 3. Deploy tự động chạy
# Xem progress trên GitHub Actions

# 4. Verify deployment
curl http://your-server:8000/health
```

---

## 📝 7. Customize CI/CD

### Thay đổi schedule training:

Edit `.github/workflows/train-model.yml`:
```yaml
schedule:
  - cron: '0 2 * * 0'  # Chủ nhật 2 AM
  # - cron: '0 0 * * *'  # Hằng ngày 12 AM
  # - cron: '0 0 * * 1'  # Thứ 2 hằng tuần
```

### Thêm notification:

Thêm vào cuối mỗi job:
```yaml
- name: Notify
  run: |
    curl -X POST YOUR_SLACK_WEBHOOK \
      -d '{"text":"Pipeline completed!"}'
```

### Skip CI cho commits nhất định:

```bash
git commit -m "docs: update README [skip ci]"
```

---

## ✅ Checklist Setup CI/CD

- [ ] Repository có code trên GitHub
- [ ] Enable GitHub Actions (Settings → Actions)
- [ ] 3 workflow files trong `.github/workflows/`
- [ ] Tests chạy được: `pytest tests/`
- [ ] Docker build được: `docker build -f docker/Dockerfile.api .`
- [ ] (Optional) Thêm secrets cho deployment
- [ ] (Optional) Cấu hình notification

---

## 🆘 Cần Giúp?

1. **Xem logs** trên GitHub Actions
2. **Test locally** trước khi push
3. **Google** error message
4. **Check** file workflow syntax

**Tips:** CI/CD giúp tự động hóa, nhưng code vẫn cần đúng! Test locally trước khi push sẽ save time. 😊

---

## 📚 Tài Liệu Thêm

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Docker Docs](https://docs.docker.com/)
- [Pytest Docs](https://docs.pytest.org/)

**Happy Coding! 🚀**
