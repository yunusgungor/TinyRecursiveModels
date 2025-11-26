#!/bin/bash

# Kubernetes Deployment Script
# Deploys the application stack to Kubernetes cluster
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

print_message "$BLUE" "🚀 Starting Kubernetes deployment..."

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_message "$RED" "❌ kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

print_message "$GREEN" "✅ kubectl is installed"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    print_message "$RED" "❌ kubectl is not configured. Please configure kubectl and try again."
    exit 1
fi

print_message "$GREEN" "✅ kubectl is configured"

# Get cluster info
print_message "$BLUE" "📊 Cluster info:"
kubectl cluster-info

# Confirm deployment
read -p "⚠️  This will deploy to the current Kubernetes cluster. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_message "$YELLOW" "❌ Deployment cancelled"
    exit 1
fi

# Create namespace
print_message "$BLUE" "📦 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Create or update secrets
print_message "$BLUE" "🔐 Setting up secrets..."
print_message "$YELLOW" "⚠️  Please ensure secrets are configured properly!"
read -p "Have you updated the secrets in k8s/secrets.yaml or created them via kubectl? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_message "$RED" "❌ Please configure secrets before deploying"
    print_message "$YELLOW" "   Option 1: Edit k8s/secrets.yaml (NOT RECOMMENDED for production)"
    print_message "$YELLOW" "   Option 2: Use kubectl create secret (RECOMMENDED)"
    print_message "$YELLOW" "   Option 3: Use external secret management (BEST for production)"
    exit 1
fi

kubectl apply -f k8s/secrets.yaml

# Apply ConfigMap
print_message "$BLUE" "⚙️  Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# Deploy PostgreSQL
print_message "$BLUE" "🗄️  Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-deployment.yaml
print_message "$BLUE" "⏳ Waiting for PostgreSQL to be ready..."
if kubectl wait --for=condition=ready pod -l app=postgres -n trendyol-gift --timeout=300s; then
    print_message "$GREEN" "✅ PostgreSQL is ready"
else
    print_message "$RED" "❌ PostgreSQL failed to start"
    kubectl logs -l app=postgres -n trendyol-gift --tail=50
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p k8s
    exit 1
fi

# Deploy Redis
print_message "$BLUE" "💾 Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml
print_message "$BLUE" "⏳ Waiting for Redis to be ready..."
if kubectl wait --for=condition=ready pod -l app=redis -n trendyol-gift --timeout=300s; then
    print_message "$GREEN" "✅ Redis is ready"
else
    print_message "$RED" "❌ Redis failed to start"
    kubectl logs -l app=redis -n trendyol-gift --tail=50
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p k8s
    exit 1
fi

# Deploy Backend
print_message "$BLUE" "🔧 Deploying Backend..."
kubectl apply -f k8s/backend-deployment.yaml
print_message "$BLUE" "⏳ Waiting for Backend to be ready..."
if kubectl wait --for=condition=ready pod -l app=backend -n trendyol-gift --timeout=300s; then
    print_message "$GREEN" "✅ Backend is ready"
else
    print_message "$RED" "❌ Backend failed to start"
    kubectl logs -l app=backend -n trendyol-gift --tail=50
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p k8s
    exit 1
fi

# Deploy Frontend
print_message "$BLUE" "🎨 Deploying Frontend..."
kubectl apply -f k8s/frontend-deployment.yaml
print_message "$BLUE" "⏳ Waiting for Frontend to be ready..."
if kubectl wait --for=condition=ready pod -l app=frontend -n trendyol-gift --timeout=300s; then
    print_message "$GREEN" "✅ Frontend is ready"
else
    print_message "$RED" "❌ Frontend failed to start"
    kubectl logs -l app=frontend -n trendyol-gift --tail=50
    print_message "$YELLOW" "⚠️  Initiating rollback..."
    ./scripts/rollback.sh -e production -p k8s
    exit 1
fi

# Deploy Ingress
print_message "$BLUE" "🌐 Deploying Ingress..."
kubectl apply -f k8s/ingress.yaml
print_message "$GREEN" "✅ Ingress deployed"

# Deploy HPA
print_message "$BLUE" "📈 Deploying Horizontal Pod Autoscaler..."
kubectl apply -f k8s/hpa.yaml
print_message "$GREEN" "✅ HPA deployed"

# Show deployment status
print_message "$GREEN" ""
print_message "$GREEN" "✅ Kubernetes deployment complete!"
print_message "$GREEN" ""

# Run deployment verification
print_message "$BLUE" "🔍 Running deployment verification..."
if ./scripts/verify-deployment.sh -e production -p k8s -r; then
    print_message "$GREEN" "✅ Deployment verification passed!"
else
    print_message "$RED" "❌ Deployment verification failed!"
    print_message "$YELLOW" "   Check logs above for details"
    exit 1
fi

print_message "$GREEN" ""
print_message "$BLUE" "📊 Deployment Status:"
kubectl get all -n trendyol-gift
print_message "$GREEN" ""
print_message "$BLUE" "🔍 Pod Status:"
kubectl get pods -n trendyol-gift
print_message "$GREEN" ""
print_message "$BLUE" "🌐 Ingress Status:"
kubectl get ingress -n trendyol-gift
print_message "$GREEN" ""
print_message "$BLUE" "📈 HPA Status:"
kubectl get hpa -n trendyol-gift
print_message "$GREEN" ""
print_message "$BLUE" "📝 Useful Commands:"
print_message "$NC" "   View logs: kubectl logs -f deployment/backend -n trendyol-gift"
print_message "$NC" "   Port forward: kubectl port-forward -n trendyol-gift svc/backend-service 8000:8000"
print_message "$NC" "   Scale: kubectl scale deployment backend --replicas=5 -n trendyol-gift"
print_message "$NC" "   Delete: kubectl delete namespace trendyol-gift"
print_message "$GREEN" ""
print_message "$BLUE" "🔄 Rollback:"
print_message "$NC" "   ./scripts/rollback.sh -e production -p k8s"
print_message "$GREEN" ""
print_message "$YELLOW" "⚠️  Next Steps:"
print_message "$NC" "   1. Configure DNS to point to the Ingress IP"
print_message "$NC" "   2. Set up SSL certificates (cert-manager)"
print_message "$NC" "   3. Configure monitoring and alerting"
print_message "$NC" "   4. Set up backup schedule"
print_message "$NC" "   5. Review and test all endpoints"
