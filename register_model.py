"""Register model to MLflow Model Registry."""
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Best model run ID
RUN_ID = "4b8e6057500b4b03bef452bac0c212dd"  # efficientnet_b0_optimized

print("=" * 80)
print("📦 REGISTERING MODEL TO MLFLOW MODEL REGISTRY")
print("=" * 80)

try:
    client = mlflow.tracking.MlflowClient()

    # Step 1: Create registered model (if not exists)
    model_name = "rice-disease-classifier"
    print(f"\n1️⃣ Creating registered model: {model_name}")

    try:
        client.create_registered_model(
            name=model_name, description="Production model for rice leaf disease classification"
        )
        print(f"✓ Created registered model: {model_name}")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"✓ Model already exists: {model_name}")
        else:
            raise e

    # Step 2: Add model version
    print(f"\n2️⃣ Adding model version from run: {RUN_ID}")
    model_uri = f"runs:/{RUN_ID}/model"

    # Check if model artifacts exist
    run = client.get_run(RUN_ID)
    print(f"✓ Found run: {run.info.run_name}")
    print(f"   Status: {run.info.status}")
    print(f"   Best Val Acc: {run.data.metrics.get('best_val_acc', 'N/A')}")

    # Note: MLflow log_model wasn't used during training
    # We need to log the model first
    print("\n⚠️  Model artifacts not logged to MLflow during training")
    print("   Need to log model checkpoint to MLflow first")

    print("\n3️⃣ Logging model checkpoint to MLflow...")
    import sys
    from pathlib import Path

    import torch

    sys.path.append("src")
    from models import create_model

    # Load model
    model_path = "models/efficientnet_b0_optimized/best_model.pth"
    model = create_model(model_name="efficientnet_b0", num_classes=6, pretrained=False)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Start a new run to log the model
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.pytorch.log_model(
            model,
            "model",
            conda_env={
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    "python=3.9",
                    "pip",
                    {
                        "pip": [
                            "torch==2.1.2",
                            "timm==0.9.12",
                            "albumentations==1.4.0",
                            "opencv-python==4.9.0.80",
                        ]
                    },
                ],
            },
        )
        print(f"✓ Model logged to MLflow run: {RUN_ID}")

    # Now create model version
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=RUN_ID)
    print(f"✓ Created model version: {mv.version}")

    # Step 3: Promote to Production
    print(f"\n4️⃣ Promoting model to Production stage...")
    client.transition_model_version_stage(name=model_name, version=mv.version, stage="Production")
    print(f"✓ Model v{mv.version} promoted to Production")

    print("\n" + "=" * 80)
    print("✓ MODEL REGISTRY SETUP COMPLETE")
    print("=" * 80)
    print(f"\nModel Details:")
    print(f"  Name: {model_name}")
    print(f"  Version: {mv.version}")
    print(f"  Stage: Production")
    print(f"  Run ID: {RUN_ID}")
    print(f"\n💡 View in MLflow UI: http://localhost:5000/#/models/{model_name}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nNote: Make sure MLflow server is running on localhost:5000")
