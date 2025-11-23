#!/bin/bash

# Trendyol Gift Recommendation System - Setup Script

set -e

echo "🚀 Trendyol Gift Recommendation System - Setup"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create environment files
echo ""
echo "📝 Setting up environment files..."

if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file"
else
    echo "⚠️  .env file already exists, skipping..."
fi

if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env file"
else
    echo "⚠️  backend/.env file already exists, skipping..."
fi

if [ ! -f frontend/.env ]; then
    cp frontend/.env.example frontend/.env
    echo "✅ Created frontend/.env file"
else
    echo "⚠️  frontend/.env file already exists, skipping..."
fi

# Install pre-commit hooks
echo ""
echo "🔧 Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found. Install it with: pip install pre-commit"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p nginx/ssl
echo "✅ Directories created"

echo ""
echo "=============================================="
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Edit .env files with your configuration"
echo "     - Set Trendyol API credentials"
echo "     - Update SECRET_KEY and JWT_SECRET_KEY"
echo "  2. Start the development environment:"
echo "     make dev"
echo "     or"
echo "     docker-compose up -d"
echo ""
echo "  3. Access the application:"
echo "     - Frontend: http://localhost:3000"
echo "     - Backend API: http://localhost:8000"
echo "     - API Docs: http://localhost:8000/docs"
echo ""
echo "For more information, see README.md"
