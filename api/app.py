"""FastAPI application for rice disease classification inference."""
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.transforms import get_val_transforms
from models import create_model

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")
PREDICTION_COUNTER = Counter("predictions_by_class", "Predictions by class", ["class_name"])

# Global variables
MODEL = None
DEVICE = None
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
]
TRANSFORMS = None


def load_model(model_path: str = "models/best_model.pth"):
    """Load trained model."""
    global MODEL, DEVICE, TRANSFORMS

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    MODEL = create_model(
        model_name=os.getenv("MODEL_NAME", "efficientnet_b0"),
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    MODEL.to(DEVICE)
    MODEL.eval()

    # Load transforms
    TRANSFORMS = get_val_transforms(image_size=int(os.getenv("IMAGE_SIZE", "224")))

    print(f"✓ Model loaded on {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    try:
        load_model(model_path)
        print("✓ Application startup complete")
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")

    yield

    # Shutdown
    print("✓ Application shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Rice Disease Classification API",
    description="API for classifying rice leaf diseases",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
]
TRANSFORMS = None


class PredictionResponse(BaseModel):
    """Prediction response model."""

    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time: float


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    model_loaded: bool
    device: str


def load_model(model_path: str = "models/best_model.pth"):
    """Load trained model."""
    global MODEL, DEVICE, TRANSFORMS

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    MODEL = create_model(
        model_name=os.getenv("MODEL_NAME", "efficientnet_b0"),
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    MODEL.to(DEVICE)
    MODEL.eval()

    # Load transforms
    TRANSFORMS = get_val_transforms(image_size=int(os.getenv("IMAGE_SIZE", "224")))

    print(f"✓ Model loaded on {DEVICE}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    try:
        load_model(model_path)
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Rice Disease Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "metrics": "/metrics",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
    )


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": MODEL.model_name,
        "num_classes": MODEL.num_classes,
        "total_parameters": MODEL.get_num_params(),
        "trainable_parameters": MODEL.get_trainable_params(),
        "device": str(DEVICE),
        "class_names": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        Prediction response
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocess
        if TRANSFORMS:
            augmented = TRANSFORMS(image=image)
            image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        else:
            raise HTTPException(status_code=500, detail="Transforms not initialized")

        # Inference
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        # Prepare response
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

        prob_dict = {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(CLASS_NAMES)
        }

        inference_time = time.time() - start_time
        REQUEST_LATENCY.observe(inference_time)
        PREDICTION_COUNTER.labels(class_name=predicted_class).inc()

        return PredictionResponse(
            class_name=predicted_class,
            confidence=confidence_score,
            probabilities=prob_dict,
            inference_time=inference_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict disease from multiple uploaded images.

    Args:
        files: List of uploaded image files

    Returns:
        List of predictions
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Preprocess
            if TRANSFORMS:
                augmented = TRANSFORMS(image=image)
                image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
            else:
                raise HTTPException(status_code=500, detail="Transforms not initialized")

            # Inference
            with torch.no_grad():
                outputs = MODEL(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)

            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item()

            results.append(
                {
                    "filename": file.filename,
                    "class_name": predicted_class,
                    "confidence": confidence_score,
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
