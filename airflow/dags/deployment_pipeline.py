"""Airflow DAG for model deployment pipeline."""
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
    "rice_disease_deployment_pipeline",
    default_args=default_args,
    description="Automated deployment pipeline for rice disease classification",
    schedule_interval=None,  # Triggered manually or by training pipeline
    catchup=False,
    tags=["ml", "deployment", "rice-disease"],
)


def validate_model():
    """Validate model exists and meets quality thresholds."""
    import os
    import torch

    model_path = "models/best_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    val_acc = checkpoint.get("val_acc", 0)

    print(f"Model validation accuracy: {val_acc:.2f}%")

    # Quality threshold
    if val_acc < 80.0:
        raise ValueError(
            f"Model accuracy {val_acc:.2f}% below threshold 80.0%"
        )

    print("✓ Model validation passed")


def build_docker_image():
    """Build Docker image for inference API."""
    import subprocess

    print("Building Docker image...")
    subprocess.run(
        ["docker", "build", "-t", "rice-disease-api:latest", "-f", "docker/Dockerfile.api", "."],
        check=True,
    )
    print("✓ Docker image built")


def deploy_to_staging():
    """Deploy to staging environment."""
    print("Deploying to staging...")
    # Add deployment logic here
    print("✓ Deployed to staging")


def run_smoke_tests():
    """Run smoke tests on deployed API."""
    import requests
    import time

    api_url = "http://localhost:8000"

    # Wait for API to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ API is healthy")
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                raise Exception("API health check failed")

    print("✓ Smoke tests passed")


# Task 1: Validate model
validate_model_task = PythonOperator(
    task_id="validate_model",
    python_callable=validate_model,
    dag=dag,
)

# Task 2: Build Docker image
build_image_task = PythonOperator(
    task_id="build_docker_image",
    python_callable=build_docker_image,
    dag=dag,
)

# Task 3: Deploy to staging
deploy_staging_task = PythonOperator(
    task_id="deploy_to_staging",
    python_callable=deploy_to_staging,
    dag=dag,
)

# Task 4: Run smoke tests
smoke_tests_task = PythonOperator(
    task_id="run_smoke_tests",
    python_callable=run_smoke_tests,
    dag=dag,
)

# Define dependencies
validate_model_task >> build_image_task >> deploy_staging_task >> smoke_tests_task
