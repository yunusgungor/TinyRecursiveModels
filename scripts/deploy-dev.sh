#!/bin/bash

# Development Deployment Script
# Deploys the application stack in development mode with hot reload
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_message "$BLUE" "🚀 Starting development deployment..."

# Load environment variables
if [ -f .env.development ]; then
    export $(cat .env.development | grep -v '^#' | xargs)
    print_message "$GREEN" "✅ Loaded development environment variables"
else
    print_message "$RED" "❌ .env.development file not found!"
    print_message "$YELLOW" "   Creating from example..."
    if [ -f .env.example ]; then
        cp .env.example .env.development
        print_message "$GREEN" "   ✅ Created .env.development from example"
        export $(cat .env.development | grep -v '^#' | xargs)
    else
        print_message "$RED" "   ❌ .env.example not found either!"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_message "$RED" "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

print_message "$GREEN" "✅ Docker is running"

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
print_message "$GREEN" "✅ BuildKit enabled"

# Stop existing containers
print_message "$BLUE" "🛑 Stopping existing containers..."
docker-compose down

# Build images with cache
print_message "$BLUE" "🔨 Building Docker images with cache..."
docker-compose build --parallel

# Start services
print_message "$BLUE" "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_message "$BLUE" "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
print_message "$BLUE" "🏥 Checking service health..."

max_retries=30
retry_count=0

# Check PostgreSQL
while [ $retry_count -lt $max_retries ]; do
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        print_message "$GREEN" "✅ PostgreSQL is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $max_retries ]; then
        print_message "$RED" "❌ PostgreSQL is not healthy"
        docker-compose logs postgres
        exit 1
    fi
    sleep 1
done

# Check Redis
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_message "$GREEN" "✅ Redis is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $max_retries ]; then
        print_message "$RED" "❌ Redis is not healthy"
        docker-compose logs redis
        exit 1
    fi
    sleep 1
done

# Check Backend
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_message "$GREEN" "✅ Backend is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $max_retries ]; then
        print_message "$YELLOW" "⚠️  Backend health check failed, but it might still be starting..."
        print_message "$YELLOW" "   Check logs with: docker-compose logs backend"
        break
    fi
    sleep 1
done

# Check Frontend
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_message "$GREEN" "✅ Frontend is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    if [ $retry_count -eq $max_retries ]; then
        print_message "$YELLOW" "⚠️  Frontend health check failed, but it might still be starting..."
        print_message "$YELLOW" "   Check logs with: docker-compose logs frontend"
        break
    fi
    sleep 1
done

print_message "$GREEN" ""
print_message "$GREEN" "✅ Development deployment complete!"
print_message "$GREEN" ""
print_message "$BLUE" "📝 Service URLs:"
print_message "$NC" "   Frontend: http://localhost:3000"
print_message "$NC" "   Backend API: http://localhost:8000"
print_message "$NC" "   API Docs: http://localhost:8000/docs"
print_message "$NC" "   PostgreSQL: localhost:5432"
print_message "$NC" "   Redis: localhost:6379"
print_message "$GREEN" ""
print_message "$BLUE" "📊 View logs:"
print_message "$NC" "   docker-compose logs -f"
print_message "$GREEN" ""
print_message "$BLUE" "🛑 Stop services:"
print_message "$NC" "   docker-compose down"
print_message "$GREEN" ""
print_message "$BLUE" "🔄 Rollback:"
print_message "$NC" "   ./scripts/rollback.sh -e development"
