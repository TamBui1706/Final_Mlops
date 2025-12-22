"""Airflow DAG for rice disease classification training pipeline."""
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "rice_disease_training_pipeline",
    default_args=default_args,
    description="Automated training pipeline for rice disease classification",
    schedule_interval="@weekly",
    catchup=False,
    tags=["ml", "training", "rice-disease"],
)


def validate_data():
    """Validate data availability and quality."""
    import os

    # Use absolute paths mounted in Airflow container
    train_dir = os.getenv("TRAIN_DIR", "/opt/airflow/train")
    val_dir = os.getenv("VAL_DIR", "/opt/airflow/validation")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Count samples
    train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
    val_samples = sum([len(files) for _, _, files in os.walk(val_dir)])

    print(f"✓ Training samples: {train_samples}")
    print(f"✓ Validation samples: {val_samples}")

    if train_samples == 0 or val_samples == 0:
        raise ValueError("No data found in train or validation directories")


def setup_dvc():
    """Setup DVC for data versioning."""
    import subprocess
    import os

    # Change to working directory where .dvc folder might exist
    os.chdir("/opt/airflow")
    
    try:
        # Check if DVC is available and initialized
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        print(f"✓ DVC version: {result.stdout.strip()}")
        
        # Try to check status
        subprocess.run(["dvc", "status"], check=False, capture_output=True)
        print("✓ DVC status checked")
    except FileNotFoundError:
        print("⚠ DVC not installed in Airflow container - skipping DVC setup")
        return
    except Exception as e:
        print(f"⚠ DVC setup skipped: {e}")
        return

    print("✓ DVC setup completed")


def notify_completion(**context):
    """Send notification on pipeline completion."""
    print("✓ Training pipeline completed successfully!")
    # Add notification logic here (email, Slack, etc.)


# Task 1: Validate data
validate_data_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

# Task 2: Setup DVC
setup_dvc_task = PythonOperator(
    task_id="setup_dvc",
    python_callable=setup_dvc,
    dag=dag,
)

# Task 3: Train model using Docker
train_model_task = DockerOperator(
    task_id="train_model",
    image="rice-disease-trainer:latest",
    api_version="auto",
    auto_remove=True,
    command="python src/train.py --epochs 50",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    mount_tmp_dir=False,
    dag=dag,
)

# Task 4: Evaluate model
evaluate_model_task = DockerOperator(
    task_id="evaluate_model",
    image="rice-disease-trainer:latest",
    api_version="auto",
    auto_remove=True,
    command="python src/evaluate.py",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    dag=dag,
)

# Task 5: Notify completion
notify_task = PythonOperator(
    task_id="notify_completion",
    python_callable=notify_completion,
    dag=dag,
)

# Define task dependencies
validate_data_task >> setup_dvc_task >> train_model_task >> evaluate_model_task >> notify_task
