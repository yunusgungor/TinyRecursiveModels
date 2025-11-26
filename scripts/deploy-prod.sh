#!/bin/bash

# Production Deployment Script
# Deploys the application stack in production mode with optimizations
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

print_message "$BLUE" "🚀 Starting production deployment..."

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
    print_message "$GREEN" "✅ Loaded production environment variables"
else
    print_message "$RED" "❌ .env.production file not found!"
    exit 1
fi

# Validate required environment variables
required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "TRENDYOL_API_KEY" "TRENDYOL_API_SECRET")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" == "CHANGE_ME_IN_PRODUCTION" ]; then
        print_message "$RED" "❌ Required environment variable $var is not set or has default value!"
        exit 1
    fi
done

print_message "$GREEN" "✅ All required environment variables are set"

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

# Build images with production target
print_message "$BLUE" "🔨 Building production Docker images..."
docker-compose -f docker-compose.prod.yml build --parallel

# Tag images with version
VERSION=${VERSION:-latest}
print_message "$BLUE" "🏷️  Tagging images with version: $VERSION"
docker tag trendyol-gift-backend:latest trendyol-gift-backend:$VERSION
docker tag trendyol-gift-frontend:latest trendyol-gift-frontend:$VERSION

# Push images to registry (if registry is configured)
if [ ! -z "$DOCKER_REGISTRY" ]; then
    print_message "$BLUE" "📤 Pushing images to registry: $DOCKER_REGISTRY"
    docker tag trendyol-gift-backend:$VERSION $DOCKER_REGISTRY/trendyol-gift-backend:$VERSION
    docker tag trendyol-gift-frontend:$VERSION $DOCKER_REGISTRY/trendyol-gift-frontend:$VERSION
    docker push $DOCKER_REGISTRY/trendyol-gift-backend:$VERSION
    docker push $DOCKER_REGISTRY/trendyol-gift-frontend:$VERSION
    print_message "$GREEN" "✅ Images pushed to registry"
fi

# Create backup of current deployment
print_message "$BLUE" "💾 Creating backup of current deployment..."
mkdir -p backups
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U postgres trendyol_gift_prod > backups/backup_$(date +%Y%m%d_%H%M%S).sql
    print_message "$GREEN" "✅ Database backup created"
fi

# Stop existing containers
print_message "$BLUE" "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start services
print_message "$BLUE" "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
print_message "$BLUE" "⏳ Waiting for services to be healthy..."
sleep 20

# Check service health
print_message "$BLUE" "🏥 Checking service health..."

max_retries=30
retry_count=0

# Check PostgreSQL
while [ $retry_count -lt $max_retries ]; do
    if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        print_message "$GREEN" "✅ PostgreSQL is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    print_message "$BLUE" "⏳ Waiting for PostgreSQL... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    print_message "$RED" "❌ PostgreSQL failed to start"
    docker-compose -f docker-compose.prod.yml logs postgres
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p compose
    exit 1
fi

# Check Redis
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli -a $REDIS_PASSWORD ping > /dev/null 2>&1; then
        print_message "$GREEN" "✅ Redis is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    print_message "$BLUE" "⏳ Waiting for Redis... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    print_message "$RED" "❌ Redis failed to start"
    docker-compose -f docker-compose.prod.yml logs redis
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p compose
    exit 1
fi

# Check Backend
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_message "$GREEN" "✅ Backend is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    print_message "$BLUE" "⏳ Waiting for Backend... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    print_message "$RED" "❌ Backend failed to start"
    docker-compose -f docker-compose.prod.yml logs backend
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p compose
    exit 1
fi

print_message "$GREEN" ""
print_message "$GREEN" "✅ Production deployment complete!"
print_message "$GREEN" ""

# Run deployment verification
print_message "$BLUE" "🔍 Running deployment verification..."
if ./scripts/verify-deployment.sh -e production -p compose -r; then
    print_message "$GREEN" "✅ Deployment verification passed!"
else
    print_message "$RED" "❌ Deployment verification failed!"
    print_message "$YELLOW" "   Check logs above for details"
    exit 1
fi

print_message "$GREEN" ""
print_message "$BLUE" "📝 Service Status:"
docker-compose -f docker-compose.prod.yml ps
print_message "$GREEN" ""
print_message "$BLUE" "📊 View logs:"
print_message "$NC" "   docker-compose -f docker-compose.prod.yml logs -f"
print_message "$GREEN" ""
print_message "$BLUE" "🛑 Stop services:"
print_message "$NC" "   docker-compose -f docker-compose.prod.yml down"
print_message "$GREEN" ""
print_message "$BLUE" "🔄 Rollback:"
print_message "$NC" "   ./scripts/rollback.sh -e production -p compose"
print_message "$GREEN" ""
print_message "$YELLOW" "⚠️  Remember to:"
print_message "$NC" "   - Configure SSL certificates"
print_message "$NC" "   - Set up monitoring and alerting"
print_message "$NC" "   - Configure backup schedule"
print_message "$NC" "   - Review security settings"
