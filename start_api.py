"""Start API server with best model."""
import os
import sys

# Set environment variables before importing app
os.environ["MODEL_PATH"] = "models/efficientnet_b0_optimized/best_model.pth"
os.environ["MODEL_NAME"] = "efficientnet_b0"
os.environ["IMAGE_SIZE"] = "224"

# Import and run
if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🚀 STARTING RICE DISEASE API")
    print("=" * 80)
    print(f"Model: {os.environ['MODEL_PATH']}")
    print(f"Server: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("=" * 80)

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
