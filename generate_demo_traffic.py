"""Auto generate traffic for demo."""
import requests
import random
import time
from pathlib import Path

API_URL = "http://localhost:8000/predict"

# Get all validation images
validation_dir = Path("validation")
all_images = list(validation_dir.glob("*/*.jpg")) + list(validation_dir.glob("*/*.JPG"))

print("=" * 60)
print("🚀 GENERATING 100 REQUESTS FOR DEMO")
print("=" * 60)

success_count = 0
error_count = 0

for i in range(100):
    try:
        # Random select image
        image_path = random.choice(all_images)
        
        # Send request
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            success_count += 1
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/100] ✓ {result['class_name']} ({result['confidence']:.2%}) - {result['inference_time']:.3f}s")
        else:
            error_count += 1
            print(f"[{i+1}/100] ✗ Error {response.status_code}")
    
    except Exception as e:
        error_count += 1
        print(f"[{i+1}/100] ✗ Exception: {e}")
    
    # Small delay
    time.sleep(0.2)

print("\n" + "=" * 60)
print(f"✅ Complete! {success_count} successful, {error_count} failed")
print("=" * 60)
print("\n📊 Now check your metrics:")
print("   1. Prometheus: http://localhost:9090")
print("      - Query: sum(inference_requests_total)")
print("      - Query: rate(inference_requests_total[1m])")
print("      - Query: histogram_quantile(0.95, inference_latency_seconds)")
print("\n   2. Grafana: http://localhost:3000 (admin/admin)")
print("      - Create dashboard with these queries")
print("=" * 60)
