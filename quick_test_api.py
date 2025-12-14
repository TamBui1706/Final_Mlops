"""Test API nhanh với curl commands."""
import json
import subprocess
from pathlib import Path

print("=" * 80)
print("🧪 TESTING RICE DISEASE API")
print("=" * 80)

# Test 1: Health check
print("\n1️⃣ Testing /health endpoint...")
try:
    result = subprocess.run(
        ["curl", "-s", "http://localhost:8000/health"], capture_output=True, text=True, shell=True
    )
    print(f"✓ Response: {result.stdout}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Model info
print("\n2️⃣ Testing /model/info endpoint...")
try:
    result = subprocess.run(
        ["curl", "-s", "http://localhost:8000/model/info"],
        capture_output=True,
        text=True,
        shell=True,
    )
    data = json.loads(result.stdout)
    print(f"✓ Model: {data.get('model_name')}")
    print(f"✓ Classes: {data.get('num_classes')}")
    print(f"✓ Parameters: {data.get('total_parameters'):,}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Prediction
print("\n3️⃣ Testing /predict endpoint...")
# Find a test image
test_images = list(Path("validation").rglob("*.jpg"))[:3]
if test_images:
    for img_path in test_images:
        true_label = img_path.parent.name
        print(f"\n📷 Image: {img_path}")
        print(f"   True label: {true_label}")

        try:
            cmd = f'curl -s -X POST "http://localhost:8000/predict" -F "file=@{img_path}"'
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            data = json.loads(result.stdout)

            predicted = data.get("class_name")
            confidence = data.get("confidence", 0) * 100
            is_correct = "✓" if predicted == true_label else "✗"

            print(f"   Predicted: {predicted} ({confidence:.1f}%) {is_correct}")
            print(f"   Inference time: {data.get('inference_time', 0):.3f}s")
        except Exception as e:
            print(f"   ✗ Error: {e}")
else:
    print("✗ No test images found in validation/")

print("\n" + "=" * 80)
print("✓ API testing complete!")
print("=" * 80)
print("\n💡 Tips:")
print("   • Open http://localhost:8000/docs for interactive API docs")
print("   • Use python test_api.py for detailed testing")
print("   • Check http://localhost:8000/metrics for Prometheus metrics")
