# 📈 Monitoring Guide - Prometheus & Grafana

Hướng dẫn setup và sử dụng monitoring stack cho Rice Disease Classification.

## 🚀 Quick Start

### 1. Khởi Động Services

```bash
# Start monitoring stack với Docker Compose
docker-compose up -d prometheus grafana

# Hoặc start tất cả services
docker-compose up -d

# Kiểm tra services đang chạy
docker ps | grep -E "prometheus|grafana"
```

### 2. Truy Cập Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Prometheus** | http://localhost:9090 | No auth |
| **Grafana** | http://localhost:3000 | admin / admin |
| **API Metrics** | http://localhost:8000/metrics | No auth |

## 📊 Prometheus - Metrics Collection

### Truy Cập Prometheus UI

```
http://localhost:9090
```

### Prometheus Queries Hữu Ích

**1. Predictions per second:**
```promql
rate(predictions_total[1m])
```

**2. P95 Latency:**
```promql
histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))
```

**3. Error Rate (%):**
```promql
rate(inference_errors_total[5m]) / rate(predictions_total[5m]) * 100
```

**4. Requests per endpoint:**
```promql
sum by(endpoint) (rate(api_requests_total[5m]))
```

**5. Top 3 predicted classes:**
```promql
topk(3, sum by(class) (rate(predictions_by_class[5m])))
```

**6. Average confidence score:**
```promql
avg(prediction_confidence)
```

### Xem Metrics Raw

```
http://localhost:8000/metrics
```

Output mẫu:
```
# HELP predictions_total Total number of predictions
# TYPE predictions_total counter
predictions_total{class="healthy"} 152.0
predictions_total{class="leaf_blast"} 89.0

# HELP inference_duration_seconds Inference duration in seconds
# TYPE inference_duration_seconds histogram
inference_duration_seconds_bucket{le="0.01"} 0.0
inference_duration_seconds_bucket{le="0.05"} 143.0
inference_duration_seconds_bucket{le="0.1"} 250.0
```

## 📈 Grafana - Visualization

### Lần Đầu Setup

1. **Truy cập Grafana:**
   ```
   http://localhost:3000
   ```

2. **Login:**
   - Username: `admin`
   - Password: `admin`
   - (Sẽ bắt đổi password lần đầu)

3. **Data Source đã auto-configured:**
   - Configuration → Data Sources
   - Prometheus data source đã sẵn

### Import Dashboard

**Cách 1: Sử dụng Dashboard có sẵn**

Dashboard đã được provisioned tự động tại:
```
monitoring/grafana/dashboards/model-performance.json
```

Truy cập: Dashboards → Browse → "Rice Disease Model Performance"

**Cách 2: Import Dashboard từ Grafana.com**

1. Dashboards → Import
2. Nhập ID: `1860` (Node Exporter) hoặc `3662` (Prometheus 2.0)
3. Select Prometheus data source
4. Click Import

### Tạo Dashboard Mới

1. Click **+** → Dashboard
2. Add → Visualization
3. Chọn Prometheus data source
4. Nhập query (ví dụ: `rate(predictions_total[1m])`)
5. Customize visualization (Graph, Gauge, Table, etc.)
6. Save Dashboard

### Dashboard Panels Khuyến Nghị

#### Panel 1: Predictions Rate
```promql
Query: rate(predictions_total[1m])
Type: Graph
Legend: Predictions/sec
```

#### Panel 2: Latency Percentiles
```promql
Query 1: histogram_quantile(0.50, rate(inference_duration_seconds_bucket[5m]))
Query 2: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))
Query 3: histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))
Type: Graph
Legend: P50, P95, P99
```

#### Panel 3: Error Rate
```promql
Query: rate(inference_errors_total[5m]) / rate(predictions_total[5m]) * 100
Type: Stat
Unit: percent
```

#### Panel 4: Class Distribution
```promql
Query: sum by(class) (predictions_by_class)
Type: Pie Chart
Legend: {{class}}
```

## 🔔 Alerts Configuration

### Tạo Alert trong Grafana

1. **Edit Panel** → Alert tab
2. Create Alert Rule:

**Alert 1: High Error Rate**
```
Condition:
  WHEN last() OF query(A, 5m, now) IS ABOVE 5

Query A: rate(inference_errors_total[5m]) / rate(predictions_total[5m]) * 100

Message: "Error rate is above 5%"
```

**Alert 2: High Latency**
```
Condition:
  WHEN last() OF query(A, 5m, now) IS ABOVE 0.5

Query A: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))

Message: "P95 latency is above 500ms"
```

### Notification Channels

**Setup Slack:**
1. Alerting → Notification channels → Add channel
2. Type: Slack
3. URL: `https://hooks.slack.com/services/YOUR/WEBHOOK/URL`
4. Channel: `#ml-alerts`
5. Test & Save

**Setup Email:**
1. Edit `monitoring/grafana/grafana.ini`:
```ini
[smtp]
enabled = true
host = smtp.gmail.com:587
user = your-email@gmail.com
password = your-app-password
from_address = alerts@example.com
```
2. Restart Grafana

## 🎯 Monitoring Best Practices

### Metrics to Monitor

**Model Performance:**
- ✅ Prediction accuracy/confidence
- ✅ Inference latency (p50, p95, p99)
- ✅ Predictions per second
- ✅ Error rate
- ✅ Class distribution

**System Health:**
- ✅ CPU usage
- ✅ Memory usage
- ✅ GPU utilization (if available)
- ✅ API request rate
- ✅ Active connections

**Business Metrics:**
- ✅ Daily active users
- ✅ Total predictions
- ✅ Disease detection frequency

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Error Rate | >2% | >5% |
| P95 Latency | >200ms | >500ms |
| CPU Usage | >70% | >90% |
| Memory Usage | >80% | >95% |
| Accuracy Drop | <93% | <90% |

## 🔍 Troubleshooting

### Prometheus không thấy metrics

**Vấn đề:** Prometheus không scrape được metrics từ API

**Giải pháp:**
```bash
# 1. Check API đang chạy và expose /metrics
curl http://localhost:8000/metrics

# 2. Check Prometheus config
cat monitoring/prometheus.yml

# 3. Check Prometheus targets
# Vào http://localhost:9090/targets
# Tất cả targets phải là "UP"

# 4. Restart Prometheus
docker-compose restart prometheus
```

### Grafana không connect được Prometheus

**Vấn đề:** "Error reading Prometheus"

**Giải pháp:**
```bash
# 1. Check Prometheus đang chạy
docker ps | grep prometheus

# 2. Check network connectivity
docker exec rice-grafana ping prometheus

# 3. Verify datasource URL
# Grafana → Configuration → Data Sources
# URL phải là: http://prometheus:9090

# 4. Test connection trong Grafana UI
```

### Không thấy data trong Dashboard

**Vấn đề:** Dashboard trống

**Giải pháp:**
```bash
# 1. Generate some traffic
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# 2. Check time range trong Grafana (top right)
# Đổi từ "Last 6 hours" → "Last 15 minutes"

# 3. Verify queries
# Click panel title → Edit → Xem query và data
```

### Container không start

**Vấn đề:** `docker-compose up -d` fail

**Giải pháp:**
```bash
# 1. Check logs
docker-compose logs prometheus
docker-compose logs grafana

# 2. Check config files syntax
promtool check config monitoring/prometheus.yml

# 3. Check ports không bị conflict
netstat -an | grep -E "3000|9090"

# 4. Remove và recreate
docker-compose down -v
docker-compose up -d
```

## 📊 Advanced Usage

### Custom Metrics trong Code

Thêm metrics mới vào API:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
custom_metric = Counter(
    'custom_predictions_total',
    'Custom prediction counter',
    ['model_version', 'device']
)

# Use in code
@app.post("/predict")
async def predict(file: UploadFile):
    result = model.predict(image)
    custom_metric.labels(
        model_version='1.0',
        device='cpu'
    ).inc()
    return result
```

### Export Metrics

```bash
# Export to JSON
curl http://localhost:9090/api/v1/query?query=predictions_total > metrics.json

# Export grafana dashboard
# Grafana → Dashboard → Settings → JSON Model → Copy
```

### Backup & Restore

**Backup Prometheus data:**
```bash
docker-compose stop prometheus
tar -czf prometheus-backup.tar.gz prometheus_data/
docker-compose start prometheus
```

**Backup Grafana dashboards:**
```bash
docker exec rice-grafana grafana-cli admin export-dashboard > dashboards-backup.json
```

## 📚 Resources

- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PromQL Guide**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Dashboard Gallery**: https://grafana.com/grafana/dashboards/

---

## 🎉 Summary

Bây giờ bạn có:

✅ **Prometheus** thu thập metrics tự động
✅ **Grafana** visualize dashboards đẹp
✅ **Alerts** cảnh báo khi có vấn đề
✅ **Monitoring** model performance real-time

**Next steps:**
1. Start services: `docker-compose up -d`
2. Open Grafana: http://localhost:3000
3. View dashboard: "Rice Disease Model Performance"
4. Setup alerts cho production

Happy Monitoring! 📈🎉
