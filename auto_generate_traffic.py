"""
Tự động generate traffic liên tục để demo Prometheus & Grafana
"""
import requests
import random
import time
from pathlib import Path
import argparse

def get_random_image():
    """Lấy 1 ảnh random từ validation set"""
    validation_path = Path("validation")
    
    if not validation_path.exists():
        print("❌ Folder validation không tồn tại!")
        return None
    
    # Lấy tất cả ảnh jpg/png
    image_files = list(validation_path.rglob("*.jpg")) + list(validation_path.rglob("*.png"))
    
    if not image_files:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    return random.choice(image_files)

def send_request(api_url="http://localhost:8000/predict"):
    """Gửi 1 request đến API"""
    try:
        image_path = get_random_image()
        if not image_path:
            return False
        
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            class_name = result.get('class_name', 'Unknown')
            confidence = result.get('confidence', 0.0)
            print(f"✓ {class_name} ({confidence:.1%}) - {image_path.name}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Auto-generate traffic for monitoring demo")
    parser.add_argument("--interval", type=float, default=2.0, 
                        help="Seconds between requests (default: 2.0)")
    parser.add_argument("--max-requests", type=int, default=0,
                        help="Maximum requests to send (0 = unlimited, default: 0)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/predict",
                        help="API endpoint URL")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 AUTO TRAFFIC GENERATOR - Rice Disease API")
    print("=" * 80)
    print(f"API URL: {args.api_url}")
    print(f"Interval: {args.interval}s between requests")
    print(f"Max requests: {'Unlimited' if args.max_requests == 0 else args.max_requests}")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 80)
    
    total_requests = 0
    success_count = 0
    
    try:
        while True:
            if send_request(args.api_url):
                success_count += 1
            
            total_requests += 1
            
            # Hiển thị progress mỗi 10 requests
            if total_requests % 10 == 0:
                success_rate = (success_count / total_requests) * 100
                print(f"\n📊 Progress: {total_requests} requests, {success_count} success ({success_rate:.1f}%)\n")
            
            # Check max requests
            if args.max_requests > 0 and total_requests >= args.max_requests:
                print(f"\n✅ Completed {total_requests} requests!")
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 STOPPED BY USER")
        print("=" * 80)
        print(f"Total requests: {total_requests}")
        print(f"Successful: {success_count}")
        print(f"Success rate: {(success_count / total_requests * 100) if total_requests > 0 else 0:.1f}%")
        print("=" * 80)

if __name__ == "__main__":
    main()
