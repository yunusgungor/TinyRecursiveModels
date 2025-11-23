#!/bin/bash

# Run all tests

set -e

echo "🧪 Running all tests..."
echo "======================"

# Backend tests
echo ""
echo "📦 Backend tests..."
cd backend
pytest -v --cov=app --cov-report=term --cov-report=html
cd ..

# Frontend tests
echo ""
echo "🎨 Frontend tests..."
cd frontend
npm test
cd ..

echo ""
echo "✅ All tests passed!"
