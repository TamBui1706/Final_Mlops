"""Register best model to MLflow Model Registry."""
import os

import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "rice-disease-classifier"
BEST_RUN_ID = "4b8e6057500b4b03bef452bac0c212dd"  # efficientnet_b0_optimized

print("=" * 80)
print("📦 MLFLOW MODEL REGISTRY SETUP")
print("=" * 80)

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

print(f"\n🔗 Connected to MLflow: {MLFLOW_URI}")

# Step 1: Get run information
print(f"\n1️⃣  Fetching run information...")
try:
    run = client.get_run(BEST_RUN_ID)
    print(f"✓ Found run: {run.info.run_name}")
    print(f"  Status: {run.info.status}")
    print(f"  Best Val Acc: {run.data.metrics.get('best_val_acc', 0):.4f}")
    print(f"  Model: {run.data.params.get('model_name', 'N/A')}")
except Exception as e:
    print(f"✗ Error fetching run: {e}")
    exit(1)

# Step 2: Create or get registered model
print(f"\n2️⃣  Setting up registered model: {MODEL_NAME}")
try:
    # Try to create
    model = client.create_registered_model(
        name=MODEL_NAME,
        description="Production model for rice leaf disease classification. "
        "Achieves 98.67% validation accuracy on 6 disease classes.",
    )
    print(f"✓ Created new registered model: {MODEL_NAME}")
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"✓ Registered model already exists: {MODEL_NAME}")
        model = client.get_registered_model(MODEL_NAME)
    else:
        print(f"✗ Error: {e}")
        exit(1)

# Step 3: Check if model artifacts exist
print(f"\n3️⃣  Checking model artifacts...")
try:
    artifacts = client.list_artifacts(BEST_RUN_ID)
    artifact_names = [a.path for a in artifacts]
    print(f"  Artifacts in run: {', '.join(artifact_names)}")

    has_model = any("model" in a for a in artifact_names)
    if not has_model:
        print(f"  ⚠️  No model artifacts found. Model was not logged to MLflow during training.")
        print(f"  ℹ️  Model checkpoint is at: models/efficientnet_b0_optimized/best_model.pth")
        print(f"\n  You can manually log it with:")
        print(f"  ```python")
        print(f"  import mlflow.pytorch")
        print(f"  with mlflow.start_run(run_id='{BEST_RUN_ID}'):")
        print(f"      mlflow.pytorch.log_model(model, 'model')")
        print(f"  ```")
        print(f"\n  For now, using checkpoint path for reference.")
except Exception as e:
    print(f"  Warning: {e}")

# Step 4: Create model version from source
print(f"\n4️⃣  Creating model version...")

# Since model wasn't logged, we'll reference the checkpoint path
source = f"file:///{os.path.abspath('models/efficientnet_b0_optimized/best_model.pth').replace(chr(92), '/')}"

try:
    # Get existing versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    next_version = len(versions) + 1

    # Create version with metadata
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{BEST_RUN_ID}/model",  # Reference to run
        run_id=BEST_RUN_ID,
        description=f"EfficientNet-B0 optimized model. "
        f"Val Acc: {run.data.metrics.get('best_val_acc', 0):.4f}. "
        f"Trained on {run.data.params.get('epochs', 'N/A')} epochs.",
    )

    print(f"✓ Created model version: {mv.version}")
    print(f"  Version: {mv.version}")
    print(f"  Current stage: {mv.current_stage}")
    print(f"  Run ID: {mv.run_id}")

except Exception as e:
    # If that fails (no artifacts), create with tags only
    print(f"  Note: {e}")
    print(f"  Creating version reference without artifacts...")

    # Add tags to run instead
    client.set_tag(BEST_RUN_ID, "model_version", "1")
    client.set_tag(BEST_RUN_ID, "model_name", MODEL_NAME)
    client.set_tag(BEST_RUN_ID, "stage", "Production")
    client.set_tag(
        BEST_RUN_ID, "checkpoint_path", "models/efficientnet_b0_optimized/best_model.pth"
    )

    print(f"✓ Tagged run with model metadata")
    mv = None

# Step 5: Promote to Production (if version was created)
if mv:
    print(f"\n5️⃣  Promoting to Production stage...")
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,  # Archive old production versions
        )
        print(f"✓ Model version {mv.version} promoted to Production")
        print(f"  Previous production versions archived")
    except Exception as e:
        print(f"✗ Error promoting: {e}")
else:
    print(f"\n5️⃣  Model version not created (no MLflow artifacts)")
    print(f"  Using run tags for tracking instead")

# Step 6: Summary
print(f"\n{'='*80}")
print(f"✓ MODEL REGISTRY SETUP COMPLETE")
print(f"{'='*80}")

print(f"\n📊 Model Details:")
print(f"  Name: {MODEL_NAME}")
print(f"  Run ID: {BEST_RUN_ID}")
print(f"  Run Name: {run.info.run_name}")
print(f"  Validation Accuracy: {run.data.metrics.get('best_val_acc', 0):.2%}")
print(f"  Model Type: {run.data.params.get('model_name', 'N/A')}")
print(f"  Checkpoint: models/efficientnet_b0_optimized/best_model.pth")

if mv:
    print(f"  Version: {mv.version}")
    print(f"  Stage: Production")
    print(f"\n🌐 View in MLflow UI:")
    print(f"  http://localhost:5000/#/models/{MODEL_NAME}")
else:
    print(f"\n🌐 View run in MLflow UI:")
    print(f"  http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{BEST_RUN_ID}")

print(f"\n💡 Usage:")
print(f"  # Load from checkpoint")
print(f"  checkpoint = torch.load('models/efficientnet_b0_optimized/best_model.pth')")
print(f"  model.load_state_dict(checkpoint['model_state_dict'])")

print(f"\n{'='*80}")

# Import os for path
import os
