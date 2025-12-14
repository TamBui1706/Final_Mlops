# 📊 Monitoring & Observability Setup Guide

## ⚠️ Vấn Đề: Empty Query Results & No Dashboard

Nếu bạn thấy "Empty query result" trong Prometheus và không có dashboard trong Grafana, đây là cách fix:

---

## 🔧 Bước 1: Verify Services Đang Chạy

```powershell
# Check containers
docker ps | Select-String "prometheus|grafana|api"

# Nếu chưa chạy, start:
docker start rice-prometheus
docker start rice-grafana

# Hoặc start toàn bộ monitoring stack:
docker-compose up -d prometheus grafana
```

---

## 🔧 Bước 2: Generate Metrics Data

**Vấn đề:** API chưa có requests nào nên metrics còn empty!

**Solution:** Gửi vài requests để tạo data:

```powershell
# Test health endpoint nhiều lần
for ($i = 1; $i -le 10; $i++) {
    curl -UseBasicParsing http://localhost:8000/health
    Start-Sleep -Milliseconds 100
}

# Test prediction với image
$image = Get-Item "validation\healthy\*.jpg" | Select-Object -First 1
curl -UseBasicParsing -Method POST -Uri "http://localhost:8000/predict" -InFile $image.FullName

# Hoặc dùng Python script
python quick_test_api.py
```

---

## 🔧 Bước 3: Verify Prometheus Scraping

### 3.1 Check Prometheus Targets

1. Mở: **http://localhost:9090**
2. Click **"Status" → "Targets"**
3. Xem target `rice-api`:
   - **State:** UP (màu xanh) ✅
   - **State:** DOWN (màu đỏ) ❌

**Nếu DOWN:**
- API không expose metrics ở `/metrics`
- Prometheus không connect được đến API
- Check network trong docker-compose

### 3.2 Test Queries Cơ Bản

Trong Prometheus (http://localhost:9090), tab "Graph":

**Query 1: Check metrics có tồn tại không**
```promql
up
```
**Expected result:** Thấy `rice-api` và `prometheus` với value = 1

**Query 2: Python metrics (luôn có)**
```promql
python_info
```
**Expected result:** Metadata về Python version

**Query 3: Request count**
```promql
inference_requests_total
```
**Expected result:** Counter tăng dần theo số requests

**Query 4: Latency**
```promql
rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])
```
**Expected result:** Average latency in seconds

---

## 🔧 Bước 4: Tạo Grafana Dashboard

Grafana chưa có dashboard vì chưa được config! Hãy tạo:

### 4.1 Login Grafana

1. Mở: **http://localhost:3000**
2. Login: `admin` / `admin`
3. (First time) Đổi password hoặc skip

### 4.2 Add Prometheus Data Source

**Nếu chưa có datasource:**

1. Click **"Connections" → "Data sources"** (hoặc biểu tượng ⚙️)
2. Click **"Add data source"**
3. Chọn **"Prometheus"**
4. Config:
   - **Name:** `Prometheus`
   - **URL:** `http://prometheus:9090` (trong Docker network)
     - Hoặc `http://localhost:9090` (nếu Grafana chạy local)
5. Click **"Save & test"**
6. Phải thấy: "Data source is working" ✅

### 4.3 Import Dashboard

**Option A: Import từ file JSON**

1. Click **"Dashboards" → "Import"** (icon +)
2. Upload file JSON dashboard (tạo bên dưới)
3. Chọn Prometheus datasource
4. Click **"Import"**

**Option B: Tạo Dashboard Mới**

1. Click **"Dashboards" → "New" → "New Dashboard"**
2. Click **"Add visualization"**
3. Chọn Prometheus datasource
4. Thêm các panels (xem bên dưới)

---

## 📊 Dashboard Panels Chi Tiết

### Panel 1: Request Rate (requests/second)

**Query:**
```promql
rate(inference_requests_total[1m])
```

**Settings:**
- **Title:** API Request Rate
- **Visualization:** Time series (Line)
- **Unit:** requests/sec (reqps)
- **Legend:** {{job}}

### Panel 2: Average Response Time

**Query:**
```promql
rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])
```

**Settings:**
- **Title:** Average Response Time
- **Visualization:** Time series
- **Unit:** seconds (s)
- **Decimals:** 3
- **Legend:** avg latency

### Panel 3: P95 Response Time

**Query:**
```promql
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))
```

**Settings:**
- **Title:** P95 Response Time
- **Visualization:** Stat (big number)
- **Unit:** seconds (s)
- **Threshold:**
  - Good: < 0.1s (green)
  - Warning: 0.1-0.5s (yellow)
  - Bad: > 0.5s (red)

### Panel 4: Total Requests Counter

**Query:**
```promql
inference_requests_total
```

**Settings:**
- **Title:** Total Requests
- **Visualization:** Stat
- **Unit:** short
- **Color mode:** Value

### Panel 5: Predictions by Class

**Query:**
```promql
rate(predictions_by_class[5m])
```

**Settings:**
- **Title:** Predictions per Class
- **Visualization:** Bar chart
- **Legend:** {{class_name}}
- **Unit:** predictions/sec

### Panel 6: Error Rate (placeholder)

**Query:**
```promql
rate(http_requests_total{status=~"5.."}[5m])
```

**Settings:**
- **Title:** Error Rate
- **Visualization:** Stat
- **Unit:** errors/sec
- **Threshold:** > 0 (red)

---

## 🎨 Complete Dashboard JSON

Save file này và import vào Grafana:

**File:** `monitoring/grafana/dashboards/rice-disease-api.json`

```json
{
  "dashboard": {
    "title": "Rice Disease API Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(inference_requests_total[1m])",
            "legendFormat": "{{job}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "fieldConfig": {
          "defaults": {
            "unit": "reqps"
          }
        }
      },
      {
        "id": 2,
        "title": "Average Response Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])",
            "legendFormat": "avg latency"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "decimals": 3
          }
        }
      },
      {
        "id": 3,
        "title": "P95 Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.1, "color": "yellow"},
                {"value": 0.5, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "Total Requests",
        "type": "stat",
        "targets": [
          {
            "expr": "inference_requests_total"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
      },
      {
        "id": 5,
        "title": "Predictions by Class",
        "type": "barchart",
        "targets": [
          {
            "expr": "rate(predictions_by_class[5m])",
            "legendFormat": "{{class_name}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "refresh": "5s",
    "time": {
      "from": "now-15m",
      "to": "now"
    }
  }
}
```

---

## 🚀 Quick Demo Scenario

### Scenario: Generate Traffic & Monitor

**Terminal 1: Generate continuous traffic**
```powershell
# Loop infinito gửi requests
while ($true) {
    curl -UseBasicParsing http://localhost:8000/health | Out-Null
    Start-Sleep -Milliseconds 500
}
```

**Terminal 2: Send prediction requests**
```powershell
# Gửi prediction requests
$images = Get-ChildItem "validation\*\*.jpg" -File | Get-Random -Count 10

foreach ($img in $images) {
    Write-Host "Testing: $($img.Name)"
    # Send request (cần implement upload)
    curl -UseBasicParsing -Method POST http://localhost:8000/predict -InFile $img.FullName
    Start-Sleep -Seconds 1
}
```

**Browser 1: Watch Prometheus (http://localhost:9090)**
```promql
# Paste queries này vào Prometheus
rate(inference_requests_total[1m])
```

**Browser 2: Watch Grafana (http://localhost:3000)**
- Xem dashboard real-time
- Numbers tăng dần
- Charts update mỗi 5 giây

---

## ❓ Troubleshooting

### Problem: "Empty query result"

**Reasons:**
1. ❌ Chưa có requests nào → **Generate traffic**
2. ❌ Prometheus chưa scrape được → **Check targets**
3. ❌ Metrics tên sai → **Check metrics list**
4. ❌ Time range quá ngắn → **Extend time range**

**Solutions:**
```powershell
# 1. Generate traffic
for ($i = 1; $i -le 50; $i++) {
    curl -UseBasicParsing http://localhost:8000/health | Out-Null
}

# 2. Check Prometheus targets
# Go to: http://localhost:9090/targets
# rice-api should be UP

# 3. List all metrics
# Go to: http://localhost:9090/api/v1/label/__name__/values
# Or query: {__name__=~".+"}

# 4. Change time range in Prometheus/Grafana
# From: now-1h → now-15m → now-5m
```

### Problem: "No dashboards found"

Grafana dashboard phải được tạo manual hoặc import!

**Quick fix:**
1. Tạo file JSON dashboard (xem trên)
2. Import vào Grafana: Dashboards → Import → Upload JSON
3. Hoặc tạo manual: New Dashboard → Add Panel

### Problem: Prometheus not scraping API

**Check:**
```powershell
# Verify API metrics endpoint
curl http://localhost:8000/metrics

# Should see output like:
# # HELP inference_requests_total
# inference_requests_total 0.0
```

**Fix prometheus config:**
```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: 'rice-api'
    static_configs:
      - targets: ['api:8000']  # Docker network
        # OR: ['host.docker.internal:8000']  # If API on host
    metrics_path: '/metrics'
```

---

## 🎯 Demo Checklist

- [ ] Prometheus running: http://localhost:9090 ✅
- [ ] Grafana running: http://localhost:3000 ✅
- [ ] API running & has `/metrics`: http://localhost:8000/metrics ✅
- [ ] Prometheus targets UP (Status → Targets) ✅
- [ ] Generated traffic (50+ requests) ✅
- [ ] Metrics visible in Prometheus (try `up` query) ✅
- [ ] Grafana datasource configured ✅
- [ ] Grafana dashboard created/imported ✅
- [ ] Dashboard shows live data ✅

---

## 💡 Pro Tips

1. **Start with simple queries:** Dùng `up` query đầu tiên để verify connection
2. **Use metric explorer:** Prometheus có autocomplete, gõ vài chữ sẽ suggest
3. **Check time range:** Nếu empty, có thể time range không match data
4. **Refresh rate:** Set dashboard refresh = 5s để xem real-time
5. **Generate load:** Trong demo, chạy loop request để chart động

---

## 🎬 Demo Script

**[Open Prometheus: http://localhost:9090]**

> "Bây giờ xem monitoring. Prometheus collect metrics từ API mỗi 15 giây."

**[Go to Status → Targets]**

> "Đây là scrape targets. rice-api đang UP, Prometheus đang scrape được metrics."

**[Go to Graph tab, query: up]**

> "Query đơn giản nhất: 'up'. Value 1 nghĩa là service healthy."

**[Query: rate(inference_requests_total[1m])]**

> "Request rate của API. Hiện tại X requests per second."

**[Open Grafana: http://localhost:3000]**

> "Grafana visualize metrics đẹp hơn. Dashboard này show request rate, response time, P95 latency..."

**[Point to charts]**

> "Nhìn đây - response time trung bình ~23ms, rất nhanh. P95 latency ~30ms. Error rate = 0. System healthy!"

**[Optional: Generate load in terminal]**

> "Nếu tôi gửi nhiều requests..."

**[Watch charts update]**

> "...charts update real-time. Request rate tăng, latency vẫn stable. System scale tốt!"

---

**Với guide này, bạn có thể setup và demo monitoring hoàn chỉnh!** 🚀
