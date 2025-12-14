# Quick start script for Windows
# Rice Disease Classification Project

Write-Host "Rice Disease Classification - Quick Start" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
} catch {
    Write-Host "Warning: Could not activate virtual environment automatically" -ForegroundColor Yellow
    Write-Host "Please run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

# Install dependencies
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
Write-Host "  - Upgrading pip and setuptools..." -ForegroundColor Gray
python -m pip install --upgrade pip setuptools wheel
Write-Host "  - Installing project dependencies..." -ForegroundColor Gray
pip install -r requirements.txt
Write-Host "  - Installing project in editable mode..." -ForegroundColor Gray
pip install -e .

# Copy environment file
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# Initialize Git repository
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    try {
        git init
        Write-Host "  Git repository initialized" -ForegroundColor Green
    } catch {
        Write-Host "Warning: Could not initialize Git repository" -ForegroundColor Yellow
    }
}

# Setup pre-commit hooks
Write-Host "Setting up pre-commit hooks..." -ForegroundColor Yellow
try {
    pre-commit install
} catch {
    Write-Host "Warning: Could not install pre-commit hooks" -ForegroundColor Yellow
}

# Initialize DVC
if (-not (Test-Path ".dvc")) {
    Write-Host "Initializing DVC..." -ForegroundColor Yellow
    try {
        dvc init
        Write-Host "  DVC initialized" -ForegroundColor Green
    } catch {
        Write-Host "Warning: Could not initialize DVC" -ForegroundColor Yellow
        Write-Host "  You can initialize it later with: dvc init" -ForegroundColor Gray
    }
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$null = New-Item -ItemType Directory -Force -Path "models"
$null = New-Item -ItemType Directory -Force -Path "logs"
$null = New-Item -ItemType Directory -Force -Path "evaluation_results"

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Place your data in train/ and validation/ folders"
Write-Host "2. Run training: python src/train.py"
Write-Host "3. Start API: python api/app.py"
Write-Host "4. View MLflow: mlflow ui"
Write-Host ""
Write-Host "Or use Docker:" -ForegroundColor Cyan
Write-Host "  docker-compose up -d"
Write-Host ""
Write-Host "Happy coding!" -ForegroundColor Green
