# Demo Docker & Containerization (Phần 8)

## 🎯 Mục tiêu Demo
Chứng minh toàn bộ hệ thống được containerize, dễ deploy, reproducible và scalable.

## 📋 Chuẩn bị

### Kiểm tra containers đang chạy
```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Containers cần thiết:**
- `rice-postgres` - Database (port 5432)
- `rice-mlflow` - MLflow server (port 5000)
- `rice-airflow-webserver` - Airflow UI (port 8080)
- `rice-airflow-scheduler` - Airflow scheduler
- `rice-prometheus` - Metrics collection (port 9090)
- `rice-grafana` - Dashboard (port 3000)
- `rice-api` - Inference API (port 8000) - có thể stopped nếu chạy local

---

## 🎬 Kịch bản Demo (5-7 phút)

### **Bước 1: Giới thiệu Docker Architecture (1 phút)**

**Hiển thị docker-compose.yml:**
```powershell
code docker-compose.yml
```

**Chỉ vào các services:**
- `postgres` - Persistent storage cho MLflow và Airflow
- `mlflow` - Tracking server với PostgreSQL backend
- `trainer` - Training service với GPU support
- `api` - REST API cho inference
- `airflow-webserver/scheduler` - Workflow orchestration
- `prometheus/grafana` - Monitoring stack

**Nói:** *"Toàn bộ hệ thống được containerize. Mỗi service chạy độc lập, dễ dàng scale và deploy."*

---

### **Bước 2: Demo Dockerfiles (2 phút)**

#### 2.1. Training Dockerfile
```powershell
code docker/Dockerfile.train
```

**Highlight các điểm:**
```dockerfile
FROM python:3.9-slim
# Install system dependencies for CV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # ... OpenCV dependencies

# Copy và install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
```

**Nói:** *"Dockerfile.train chứa tất cả dependencies để train model. Image này portable, chạy được ở bất kỳ đâu có Docker."*

#### 2.2. API Dockerfile
```powershell
code docker/Dockerfile.api
```

**Highlight:**
```dockerfile
# Lightweight Python base
FROM python:3.9-slim

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import requests; \
        requests.get('http://localhost:8000/health')"

# Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Nói:** *"API Dockerfile có health check tự động. Kubernetes sẽ restart container nếu health check fail."*

---

### **Bước 3: Demo Docker Commands (2-3 phút)**

#### 3.1. Xem containers đang chạy
```powershell
docker ps
```

**Giải thích output:**
- **NAMES** - Tên container (rice-*)
- **STATUS** - Up X minutes/hours
- **PORTS** - Port mapping (host:container)

#### 3.2. Xem logs container
```powershell
# Logs của API
docker logs rice-api --tail 20

# Follow logs real-time
docker logs -f rice-prometheus
```

**Nói:** *"Mọi logs centralized. Dễ dàng debug khi có vấn đề."*

#### 3.3. Inspect container
```powershell
docker inspect rice-postgres | Select-String "IPAddress"
```

**Chỉ vào:**
- IP address trong Docker network
- Volume mounts
- Environment variables

#### 3.4. Exec vào container
```powershell
# Vào PostgreSQL container
docker exec -it rice-postgres psql -U postgres -d mlflow

# List tables trong MLflow database
\dt
\q
```

**Nói:** *"Có thể truy cập trực tiếp vào containers để debug hoặc inspect data."*

---

### **Bước 4: Demo Docker Compose (2 phút)**

#### 4.1. Kiểm tra services
```powershell
docker-compose ps
```

**Highlight:**
- Tất cả services và status
- Dependencies (depends_on)

#### 4.2. Demo restart service
```powershell
# Restart Prometheus để load config mới
docker-compose restart prometheus

# Check logs sau khi restart
docker logs rice-prometheus --tail 10
```

**Nói:** *"Docker Compose quản lý multi-container apps. Một lệnh để start/stop/restart toàn bộ stack."*

#### 4.3. Demo scale service (nếu có thời gian)
```powershell
# Scale API service lên 3 instances
docker-compose up -d --scale api=3

# Xem các instances
docker ps | Select-String "rice-api"
```

**Nói:** *"Dễ dàng scale horizontal. Thêm load balancer để distribute traffic."*

---

### **Bước 5: Demo Networking (1 phút)**

#### 5.1. Xem Docker networks
```powershell
docker network ls
docker network inspect riceleafsdisease_rice-network
```

**Giải thích:**
- **Bridge network** - Tất cả containers trong cùng network
- **Service discovery** - Containers gọi nhau bằng tên (mlflow:5000, postgres:5432)
- **Isolation** - Network isolated với host và external

**Nói:** *"Containers communicate qua internal network. Không expose ports không cần thiết ra ngoài."*

---

### **Bước 6: Demo Volumes & Persistence (1 phút)**

#### 6.1. Xem volumes
```powershell
docker volume ls
docker volume inspect riceleafsdisease_postgres_data
```

**Giải thích:**
- **postgres_data** - Database files persistent
- **mlflow_data** - MLflow artifacts persistent
- **prometheus_data** - Metrics history
- **grafana_data** - Dashboards & datasources

**Nói:** *"Data được lưu trong volumes. Khi restart containers, data không bị mất."*

#### 6.2. Demo backup volume (optional)
```powershell
# Backup PostgreSQL volume
docker run --rm \
    -v riceleafsdisease_postgres_data:/data \
    -v ${PWD}:/backup \
    alpine tar czf /backup/postgres_backup.tar.gz /data
```

---

### **Bước 7: Demo Build & Deploy Workflow (1 phút)**

#### 7.1. Build Docker images
```powershell
# Build API image
docker build -t rice-disease-api:latest -f docker/Dockerfile.api .

# Build training image
docker build -t rice-disease-trainer:latest -f docker/Dockerfile.train .
```

**Nói:** *"CI/CD pipeline tự động build images khi có code mới. Tag với version hoặc git commit hash."*

#### 7.2. Push to Registry (giải thích, không chạy)
```powershell
# Tag cho registry
docker tag rice-disease-api:latest myregistry.azurecr.io/rice-disease-api:v1.0.0

# Push lên Azure Container Registry
docker push myregistry.azurecr.io/rice-disease-api:v1.0.0
```

**Nói:** *"Production deploy: push images lên registry (Docker Hub, ACR, ECR), rồi pull từ Kubernetes cluster."*

---

### **Bước 8: Demo Health Checks (1 phút)**

#### 8.1. Check container health
```powershell
# Xem health status
docker inspect rice-api --format='{{json .State.Health}}' | ConvertFrom-Json

# Test health endpoint
curl http://localhost:8000/health
```

**Nói:** *"Health checks đảm bảo service hoạt động đúng. Auto-restart khi unhealthy."*

---

### **Bước 9: Demo Resource Management (optional)**

#### 9.1. Xem resource usage
```powershell
docker stats --no-stream
```

**Highlight:**
- CPU usage per container
- Memory usage
- Network I/O

**Nói:** *"Monitor resource usage. Set limits trong docker-compose.yml để prevent resource starvation."*

#### 9.2. Set resource limits (chỉ code)
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

---

### **Bước 10: Tổng kết (30s)**

**Điểm nhấn:**
- ✅ **Reproducibility** - Same environment everywhere (dev/staging/prod)
- ✅ **Isolation** - Mỗi service độc lập, không conflict dependencies
- ✅ **Scalability** - Dễ dàng scale horizontal
- ✅ **Portability** - Deploy anywhere có Docker (cloud, on-premise)
- ✅ **Consistency** - "Works on my machine" không còn tồn tại
- ✅ **Easy rollback** - Rollback về version cũ chỉ cần change image tag

**Nói:** *"Docker giải quyết dependency hell. Toàn bộ stack chạy consistent từ laptop đến production. Một lệnh docker-compose up để start entire MLOps platform."*

---

## 🎯 Q&A Thường gặp

### Q1: "Docker khác gì Virtual Machine?"
**A:**
- **Docker** - Share OS kernel, lightweight, start nhanh (seconds)
- **VM** - Full OS per VM, heavy, start chậm (minutes)
- **Docker** tốt cho microservices, **VM** tốt cho multi-tenancy

### Q2: "Production có dùng Docker Compose không?"
**A:**
- Dev/Staging - Docker Compose OK
- Production - Dùng orchestration platform (Kubernetes, Docker Swarm, ECS)
- Kubernetes provides: auto-scaling, self-healing, rolling updates, service mesh

### Q3: "Làm sao để secure containers?"
**A:**
- Scan images cho vulnerabilities (Trivy, Snyk)
- Use official base images
- Run non-root user
- Limit resources
- Network segmentation
- Secret management (Vault, k8s Secrets)

### Q4: "GPU training trong Docker sao?"
**A:**
```yaml
services:
  trainer:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
Requires: nvidia-docker runtime installed

### Q5: "Làm sao để debug container đang chạy?"
**A:**
```powershell
# View logs
docker logs -f container_name

# Exec shell
docker exec -it container_name bash

# Copy files ra ngoài
docker cp container_name:/app/logs ./logs

# Inspect process
docker top container_name
```

---

## 📊 Metrics để Demo

### Container Health Status
```powershell
docker ps --format "{{.Names}}: {{.Status}}"
```

### Resource Usage
```powershell
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Image Sizes
```powershell
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

---

## 🚀 Tips cho Demo mượt mà

1. **Chuẩn bị trước:**
   - All containers running (`docker-compose up -d`)
   - Build all images trước
   - Có backup commands trong file .txt

2. **Trong lúc demo:**
   - Giữ terminal output clean (clear screen trước khi chạy command)
   - Highlight key information trong output
   - Giải thích WHAT và WHY, không chỉ HOW

3. **Visual aids:**
   - Mở Docker Desktop (nếu có) để show GUI
   - Draw architecture diagram showing containers
   - Show docker-compose.yml trong VS Code với syntax highlighting

4. **Backup plan:**
   - Nếu container fail to start, show logs (`docker logs`)
   - Use as example để demo troubleshooting
   - Screenshot containers chạy tốt sẵn

---

## ✅ Checklist trước khi Demo

- [ ] All containers running: `docker ps`
- [ ] No containers in restart loop: `docker ps -a`
- [ ] Images built: `docker images | Select-String rice`
- [ ] Volumes exist: `docker volume ls`
- [ ] Network exists: `docker network ls`
- [ ] Health checks passing: `docker inspect --format='{{.State.Health.Status}}' rice-api`
- [ ] Docker Desktop running (if available)
- [ ] Đã test all demo commands

---

## 🔗 Quick Demo Commands

```powershell
# Start all services
docker-compose up -d

# View running containers
docker ps

# View logs
docker logs -f rice-api

# Exec into container
docker exec -it rice-postgres psql -U postgres -d mlflow

# Restart service
docker-compose restart prometheus

# View stats
docker stats --no-stream

# Inspect network
docker network inspect riceleafsdisease_rice-network

# Stop all
docker-compose down

# Remove all (including volumes) - DANGEROUS
docker-compose down -v
```

---

**Thời gian demo**: 5-7 phút
**Độ khó**: Trung bình
**Impact**: Cao - Chứng minh production-ready deployment strategy
