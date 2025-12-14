"""
View and analyze training comparison results
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="View model comparison results")
    parser.add_argument("--file", type=str, help="Specific results file to view")
    parser.add_argument("--latest", action="store_true", help="View latest results")
    args = parser.parse_args()

    results_dir = Path("evaluation_results")

    if args.file:
        results_path = Path(args.file)
    elif args.latest:
        # Find latest comparison file
        csv_files = list(results_dir.glob("model_comparison_*.csv"))
        if not csv_files:
            print("❌ No comparison results found in evaluation_results/")
            return
        results_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    else:
        # List all available files
        csv_files = sorted(
            results_dir.glob("model_comparison_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not csv_files:
            print("❌ No comparison results found in evaluation_results/")
            return

        print("\n📊 Available comparison results:")
        print("-" * 80)
        for i, f in enumerate(csv_files, 1):
            mtime = f.stat().st_mtime
            from datetime import datetime

            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. {f.name} (Modified: {date_str})")
        print("-" * 80)

        # Use the latest by default
        results_path = csv_files[0]
        print(f"\n📄 Loading: {results_path.name}")

    # Load and display results
    df = pd.read_csv(results_path)

    print("\n" + "=" * 120)
    print("MODEL COMPARISON RESULTS")
    print("=" * 120)
    print(f"\n{df.to_string(index=False)}\n")

    # Print winner
    best_config = df.iloc[0]
    print("=" * 120)
    print("🏆 BEST CONFIGURATION")
    print("=" * 120)
    print(f"Config Name: {best_config['config_name']}")
    print(f"Model: {best_config['model_name']}")
    print(f"Description: {best_config['description']}")
    print(f"Best Val Accuracy: {best_config['best_val_acc']:.2f}%")
    print(f"Best Val Loss: {best_config['best_val_loss']:.4f}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Total Parameters: {best_config['total_params']:,}")
    print(f"Model saved at: {best_config['checkpoint_path']}")
    print(f"MLflow Run ID: {best_config['run_id']}")
    print("=" * 120)

    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 120)
    print(f"Average Validation Accuracy: {df['best_val_acc'].mean():.2f}%")
    print(f"Std Validation Accuracy: {df['best_val_acc'].std():.2f}%")
    print(f"Best Accuracy: {df['best_val_acc'].max():.2f}%")
    print(f"Worst Accuracy: {df['best_val_acc'].min():.2f}%")
    print(f"Accuracy Range: {df['best_val_acc'].max() - df['best_val_acc'].min():.2f}%")
    print("-" * 120)

    print(f"\n✓ Loaded from: {results_path}")


if __name__ == "__main__":
    main()
