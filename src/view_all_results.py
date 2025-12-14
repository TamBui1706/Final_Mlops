"""View all training results from MLflow."""
from pathlib import Path

import mlflow
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("rice-disease-classification")
if not experiment:
    print("❌ Experiment not found!")
    exit(1)

# Search all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    filter_string="attributes.status = 'FINISHED'",  # Only successful runs
)

if runs.empty:
    print("❌ No runs found!")
    exit(1)

print(f"\n{'='*120}")
print("ALL TRAINING RUNS")
print(f"{'='*120}\n")

# Extract relevant columns
results = []
for _, run in runs.iterrows():
    config_name = run.get("tags.mlflow.runName", "Unknown")

    # Extract config name from run name (remove timestamp)
    config_name_clean = "_".join(config_name.split("_")[:-2]) if "_" in config_name else config_name

    result = {
        "config_name": config_name_clean,
        "run_name": config_name,
        "model_name": run.get("params.model_name", "N/A"),
        "learning_rate": float(run.get("params.learning_rate", 0)),
        "batch_size": int(run.get("params.batch_size", 0)),
        "weight_decay": float(run.get("params.weight_decay", 0)),
        "best_val_acc": run.get("metrics.best_val_acc", 0.0),
        "best_val_loss": run.get("metrics.best_val_loss", 0.0),
        "final_train_acc": run.get("metrics.train_acc", 0.0),
        "final_train_loss": run.get("metrics.train_loss", 0.0),
        "epochs": int(run.get("params.epochs", 0)),
        "run_id": run["run_id"],
        "status": run["status"],
    }
    results.append(result)

# Create DataFrame
df = pd.DataFrame(results)

# Filter out runs without metrics
df = df[df["best_val_acc"] > 0]

if df.empty:
    print("❌ No completed runs with metrics found!")
    exit(1)

# Group by config and get best run for each
df_best = df.loc[df.groupby("config_name")["best_val_acc"].idxmax()]
df_best = df_best.sort_values("best_val_acc", ascending=False)

print(
    df_best[
        [
            "config_name",
            "model_name",
            "learning_rate",
            "batch_size",
            "best_val_acc",
            "best_val_loss",
            "epochs",
            "status",
        ]
    ].to_string(index=False)
)

print(f"\n{'='*120}")
print("🏆 BEST MODEL")
print(f"{'='*120}")

best = df_best.iloc[0]
print(f"Config: {best['config_name']}")
print(f"Model: {best['model_name']}")
print(f"Best Val Accuracy: {best['best_val_acc']:.2f}%")
print(f"Best Val Loss: {best['best_val_loss']:.4f}")
print(f"Learning Rate: {best['learning_rate']}")
print(f"Batch Size: {best['batch_size']}")
print(f"Run ID: {best['run_id']}")

print(f"\n{'='*120}")
print("COMPARISON")
print(f"{'='*120}")
print(f"Average Accuracy: {df_best['best_val_acc'].mean():.2f}%")
print(f"Std Accuracy: {df_best['best_val_acc'].std():.2f}%")
print(f"Best: {df_best['best_val_acc'].max():.2f}%")
print(f"Worst: {df_best['best_val_acc'].min():.2f}%")
print(f"Range: {df_best['best_val_acc'].max() - df_best['best_val_acc'].min():.2f}%")

# Save comprehensive results
output_path = Path("evaluation_results") / "all_runs_comparison.csv"
output_path.parent.mkdir(exist_ok=True)
df_best.to_csv(output_path, index=False)
print(f"\n✓ Full results saved to: {output_path}")
