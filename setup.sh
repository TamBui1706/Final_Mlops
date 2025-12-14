#!/bin/bash
# Quick start script for Rice Disease Classification Project

set -e

echo "🌾 Rice Disease Classification - Quick Start"
echo "============================================="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate  # Linux/Mac
# For Windows: venv\Scripts\activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
fi

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Initialize DVC
if [ ! -d ".dvc" ]; then
    echo "📊 Initializing DVC..."
    dvc init
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs evaluation_results

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your data in train/ and validation/ folders"
echo "2. Run training: python src/train.py"
echo "3. Start API: python api/app.py"
echo "4. View MLflow: mlflow ui"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "Happy coding! 🚀"
