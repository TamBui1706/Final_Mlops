"""Generate continuous traffic to API for monitoring demo."""
import random
import time
from pathlib import Path

import requests

API_URL = "http://localhost:8000"


def get_random_images(num=10):
    """Get random images from validation set."""
    validation_path = Path("validation")
    all_images = list(validation_path.glob("*/*.jpg"))
    return random.sample(all_images, min(num, len(all_images)))


def send_prediction(image_path):
    """Send prediction request."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            # API response format: {class_name, confidence, probabilities, inference_time}
            print(
                f"✓ {image_path.name}: {result['class_name']} "
                f"({result['confidence']:.2%}) - {result['inference_time']:.3f}s"
            )
            return True
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def generate_traffic(num_requests=20, delay=1.0):
    """Generate traffic continuously."""
    print(f"🚀 Starting traffic generation...")
    print(f"📊 Target: {num_requests} requests")
    print(f"⏱️  Delay: {delay}s between requests\n")

    images = get_random_images(num_requests)
    success_count = 0

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{num_requests}] ", end="")

        if send_prediction(image_path):
            success_count += 1

        if i < num_requests:
            time.sleep(delay)

    print(f"\n✅ Complete! {success_count}/{num_requests} successful")
    print(f"\n📈 Check metrics:")
    print(f"   - Prometheus: http://localhost:9090")
    print(f"   - Grafana: http://localhost:3000")


def continuous_traffic(interval=2.0):
    """Generate traffic continuously until interrupted."""
    print("🔄 Continuous traffic mode (Ctrl+C to stop)")
    print(f"⏱️  Interval: {interval}s\n")

    try:
        count = 0
        while True:
            count += 1
            images = get_random_images(1)
            print(f"[{count}] ", end="")
            send_prediction(images[0])
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopped after {count} requests")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🎯 API Traffic Generator")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Generate 20 requests (default)")
    print("  2. Generate 50 requests")
    print("  3. Generate 100 requests")
    print("  4. Continuous mode (until Ctrl+C)")
    print()

    choice = input("Choose option (1-4) [1]: ").strip() or "1"

    if choice == "1":
        generate_traffic(num_requests=20, delay=0.5)
    elif choice == "2":
        generate_traffic(num_requests=50, delay=0.5)
    elif choice == "3":
        generate_traffic(num_requests=100, delay=0.3)
    elif choice == "4":
        continuous_traffic(interval=1.0)
    else:
        print("Invalid option!")
        sys.exit(1)
