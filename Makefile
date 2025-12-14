# Makefile for Rice Disease Classification Project

.PHONY: help install setup clean test lint format docker-build docker-up docker-down train predict api mlflow

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Setup project (venv, deps, pre-commit)"
	@echo "  make clean         - Clean cache and build files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make train         - Train model"
	@echo "  make evaluate      - Evaluate model"
	@echo "  make predict       - Run prediction"
	@echo "  make api           - Start FastAPI server"
	@echo "  make mlflow        - Start MLflow UI"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

setup:
	python -m venv venv
	.\venv\Scripts\activate
	$(MAKE) install
	pre-commit install
	@echo "Setup complete! Activate venv: .\venv\Scripts\activate"

clean:
	rm -rf __pycache__ **/__pycache__ .pytest_cache **/.pytest_cache
	rm -rf build dist *.egg-info **/*.egg-info
	rm -rf .coverage htmlcov
	rm -rf logs mlruns
	find . -name "*.pyc" -delete

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ api/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	black --check src/ api/ tests/
	isort --check-only src/ api/ tests/

format:
	black src/ api/ tests/
	isort src/ api/ tests/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services started:"
	@echo "  API: http://localhost:8000"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Airflow: http://localhost:8080"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

train:
	python src/train.py --train-dir train --val-dir validation --epochs 50

evaluate:
	python src/evaluate.py --val-dir validation --model-path models/best_model.pth

predict:
	@echo "Usage: make predict IMAGE=path/to/image.jpg"
	python src/predict.py --image $(IMAGE) --model models/best_model.pth

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --port 5000

dvc-init:
	dvc init
	dvc remote add -d myremote /path/to/storage

dvc-add:
	dvc add train validation
	git add train.dvc validation.dvc .gitignore
	git commit -m "Track data with DVC"

dvc-push:
	dvc push

dvc-pull:
	dvc pull
