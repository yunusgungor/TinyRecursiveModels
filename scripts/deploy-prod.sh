#!/bin/bash

# Production Deployment Script
set -e

echo "🚀 Starting production deployment..."

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
    echo "✅ Loaded production environment variables"
else
    echo "❌ .env.production file not found!"
    exit 1
fi

# Validate required environment variables
required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "TRENDYOL_API_KEY" "TRENDYOL_API_SECRET")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" == "CHANGE_ME_IN_PRODUCTION" ]; then
        echo "❌ Required environment variable $var is not set or has default value!"
        exit 1
    fi
done

echo "✅ All required environment variables are set"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Build images with production target
echo "🔨 Building production Docker images..."
docker-compose -f docker-compose.prod.yml build

# Tag images with version
VERSION=${VERSION:-latest}
echo "🏷️  Tagging images with version: $VERSION"
docker tag trendyol-gift-backend:latest trendyol-gift-backend:$VERSION
docker tag trendyol-gift-frontend:latest trendyol-gift-frontend:$VERSION

# Push images to registry (if registry is configured)
if [ ! -z "$DOCKER_REGISTRY" ]; then
    echo "📤 Pushing images to registry: $DOCKER_REGISTRY"
    docker tag trendyol-gift-backend:$VERSION $DOCKER_REGISTRY/trendyol-gift-backend:$VERSION
    docker tag trendyol-gift-frontend:$VERSION $DOCKER_REGISTRY/trendyol-gift-frontend:$VERSION
    docker push $DOCKER_REGISTRY/trendyol-gift-backend:$VERSION
    docker push $DOCKER_REGISTRY/trendyol-gift-frontend:$VERSION
    echo "✅ Images pushed to registry"
fi

# Create backup of current deployment
echo "💾 Creating backup of current deployment..."
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U postgres trendyol_gift_prod > backup_$(date +%Y%m%d_%H%M%S).sql
    echo "✅ Database backup created"
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start services
echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 20

# Check service health
echo "🏥 Checking service health..."

max_retries=30
retry_count=0

# Check PostgreSQL
while [ $retry_count -lt $max_retries ]; do
    if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo "✅ PostgreSQL is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "⏳ Waiting for PostgreSQL... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ PostgreSQL failed to start"
    docker-compose -f docker-compose.prod.yml logs postgres
    exit 1
fi

# Check Redis
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli -a $REDIS_PASSWORD ping > /dev/null 2>&1; then
        echo "✅ Redis is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "⏳ Waiting for Redis... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ Redis failed to start"
    docker-compose -f docker-compose.prod.yml logs redis
    exit 1
fi

# Check Backend
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "⏳ Waiting for Backend... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ Backend failed to start"
    docker-compose -f docker-compose.prod.yml logs backend
    exit 1
fi

echo ""
echo "✅ Production deployment complete!"
echo ""
echo "📝 Service Status:"
docker-compose -f docker-compose.prod.yml ps
echo ""
echo "📊 View logs:"
echo "   docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose -f docker-compose.prod.yml down"
echo ""
echo "⚠️  Remember to:"
echo "   - Configure SSL certificates"
echo "   - Set up monitoring and alerting"
echo "   - Configure backup schedule"
echo "   - Review security settings"
