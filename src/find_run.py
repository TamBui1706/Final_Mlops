"""Find specific run by name."""
import sys

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

target_name = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b0_baseline_20251214_000508"

experiment = mlflow.get_experiment_by_name("rice-disease-classification")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=100
)

print(f"\n{'='*100}")
print(f"Searching for run: {target_name}")
print(f"{'='*100}\n")

found = False
for _, run in runs.iterrows():
    run_name = run.get("tags.mlflow.runName", "")
    if target_name in run_name:
        found = True
        print(f"✓ Found: {run_name}")
        print(f"  Run ID: {run['run_id']}")
        print(f"  Status: {run['status']}")
        print(f"  Model: {run.get('params.model_name', 'N/A')}")
        print(f"  Best Val Acc: {run.get('metrics.best_val_acc', 0):.4f}")
        print(f"  Best Val Loss: {run.get('metrics.best_val_loss', 0):.4f}")
        print(f"  Final Train Acc: {run.get('metrics.train_acc', 0):.4f}")
        print(f"  Epochs: {run.get('params.epochs', 'N/A')}")
        print(f"  Batch Size: {run.get('params.batch_size', 'N/A')}")
        print(f"  Learning Rate: {run.get('params.learning_rate', 'N/A')}")
        print()

if not found:
    print(f"❌ Run not found: {target_name}")
    print("\nAvailable baseline runs:")
    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", "")
        if "efficientnet_b0_baseline" in run_name:
            print(
                f"  - {run_name} ({run['status']}) - Acc: {run.get('metrics.best_val_acc', 0):.2%}"
            )
