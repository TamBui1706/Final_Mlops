"""Test metrics endpoint."""
import requests

# Test metrics endpoint
response = requests.get("http://localhost:8000/metrics")

print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print("\nMetrics (first 500 chars):")
print(response.text[:500])

# Check if inference_requests_total exists
if "inference_requests_total" in response.text:
    print("\n✓ inference_requests_total found!")
    lines = [line for line in response.text.split('\n') if 'inference_requests_total' in line and not line.startswith('#')]
    for line in lines:
        print(f"  {line}")
else:
    print("\n✗ inference_requests_total NOT found!")
