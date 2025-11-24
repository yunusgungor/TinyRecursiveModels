#!/bin/bash

# Kubernetes Deployment Script
set -e

echo "🚀 Starting Kubernetes deployment..."

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

echo "✅ kubectl is installed"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ kubectl is not configured. Please configure kubectl and try again."
    exit 1
fi

echo "✅ kubectl is configured"

# Get cluster info
echo "📊 Cluster info:"
kubectl cluster-info

# Confirm deployment
read -p "⚠️  This will deploy to the current Kubernetes cluster. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Create namespace
echo "📦 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Create or update secrets
echo "🔐 Setting up secrets..."
echo "⚠️  Please ensure secrets are configured properly!"
read -p "Have you updated the secrets in k8s/secrets.yaml or created them via kubectl? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Please configure secrets before deploying"
    echo "   Option 1: Edit k8s/secrets.yaml (NOT RECOMMENDED for production)"
    echo "   Option 2: Use kubectl create secret (RECOMMENDED)"
    echo "   Option 3: Use external secret management (BEST for production)"
    exit 1
fi

kubectl apply -f k8s/secrets.yaml

# Apply ConfigMap
echo "⚙️  Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# Deploy PostgreSQL
echo "🗄️  Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-deployment.yaml
echo "⏳ Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n trendyol-gift --timeout=300s
echo "✅ PostgreSQL is ready"

# Deploy Redis
echo "💾 Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml
echo "⏳ Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n trendyol-gift --timeout=300s
echo "✅ Redis is ready"

# Deploy Backend
echo "🔧 Deploying Backend..."
kubectl apply -f k8s/backend-deployment.yaml
echo "⏳ Waiting for Backend to be ready..."
kubectl wait --for=condition=ready pod -l app=backend -n trendyol-gift --timeout=300s
echo "✅ Backend is ready"

# Deploy Frontend
echo "🎨 Deploying Frontend..."
kubectl apply -f k8s/frontend-deployment.yaml
echo "⏳ Waiting for Frontend to be ready..."
kubectl wait --for=condition=ready pod -l app=frontend -n trendyol-gift --timeout=300s
echo "✅ Frontend is ready"

# Deploy Ingress
echo "🌐 Deploying Ingress..."
kubectl apply -f k8s/ingress.yaml
echo "✅ Ingress deployed"

# Deploy HPA
echo "📈 Deploying Horizontal Pod Autoscaler..."
kubectl apply -f k8s/hpa.yaml
echo "✅ HPA deployed"

# Show deployment status
echo ""
echo "✅ Kubernetes deployment complete!"
echo ""
echo "📊 Deployment Status:"
kubectl get all -n trendyol-gift
echo ""
echo "🔍 Pod Status:"
kubectl get pods -n trendyol-gift
echo ""
echo "🌐 Ingress Status:"
kubectl get ingress -n trendyol-gift
echo ""
echo "📈 HPA Status:"
kubectl get hpa -n trendyol-gift
echo ""
echo "📝 Useful Commands:"
echo "   View logs: kubectl logs -f deployment/backend -n trendyol-gift"
echo "   Port forward: kubectl port-forward -n trendyol-gift svc/backend-service 8000:8000"
echo "   Scale: kubectl scale deployment backend --replicas=5 -n trendyol-gift"
echo "   Delete: kubectl delete namespace trendyol-gift"
echo ""
echo "⚠️  Next Steps:"
echo "   1. Configure DNS to point to the Ingress IP"
echo "   2. Set up SSL certificates (cert-manager)"
echo "   3. Configure monitoring and alerting"
echo "   4. Set up backup schedule"
echo "   5. Review and test all endpoints"
