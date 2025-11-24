# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Trendyol Gift Recommendation application to a production Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Docker images built and pushed to a container registry
- Persistent storage provisioner configured
- Ingress controller (nginx-ingress recommended)
- cert-manager for SSL certificates (optional but recommended)

## Directory Structure

```
k8s/
├── namespace.yaml              # Namespace definition
├── configmap.yaml             # Application configuration
├── secrets.yaml               # Sensitive data (DO NOT commit real secrets!)
├── postgres-deployment.yaml   # PostgreSQL database
├── redis-deployment.yaml      # Redis cache
├── backend-deployment.yaml    # Backend API service
├── frontend-deployment.yaml   # Frontend web application
├── ingress.yaml              # Ingress rules for external access
├── hpa.yaml                  # Horizontal Pod Autoscaler
└── README.md                 # This file
```

## Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Update Secrets

**IMPORTANT**: Before deploying, update `secrets.yaml` with your actual credentials:

```bash
# Option 1: Edit the file directly (NOT RECOMMENDED for production)
vim secrets.yaml

# Option 2: Create secrets from command line (RECOMMENDED)
kubectl create secret generic trendyol-gift-secrets \
  --namespace=trendyol-gift \
  --from-literal=POSTGRES_USER=postgres \
  --from-literal=POSTGRES_PASSWORD=your-secure-password \
  --from-literal=REDIS_PASSWORD=your-redis-password \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=TRENDYOL_API_KEY=your-api-key \
  --from-literal=TRENDYOL_API_SECRET=your-api-secret

# Option 3: Use external secret management (BEST for production)
# - AWS Secrets Manager
# - HashiCorp Vault
# - Google Secret Manager
```

### 3. Update ConfigMap

Review and update `configmap.yaml` if needed:

```bash
kubectl apply -f configmap.yaml
```

### 4. Deploy Database and Cache

```bash
# Deploy PostgreSQL
kubectl apply -f postgres-deployment.yaml

# Deploy Redis
kubectl apply -f redis-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n trendyol-gift --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n trendyol-gift --timeout=300s
```

### 5. Deploy Backend

Before deploying, ensure your Docker image is built and pushed:

```bash
# Build and push backend image
docker build -t your-registry/trendyol-gift-backend:latest -f backend/Dockerfile backend/
docker push your-registry/trendyol-gift-backend:latest

# Update image in backend-deployment.yaml
# Then deploy
kubectl apply -f backend-deployment.yaml

# Wait for backend to be ready
kubectl wait --for=condition=ready pod -l app=backend -n trendyol-gift --timeout=300s
```

### 6. Deploy Frontend

```bash
# Build and push frontend image
docker build -t your-registry/trendyol-gift-frontend:latest -f frontend/Dockerfile frontend/
docker push your-registry/trendyol-gift-frontend:latest

# Update image in frontend-deployment.yaml
# Then deploy
kubectl apply -f frontend-deployment.yaml

# Wait for frontend to be ready
kubectl wait --for=condition=ready pod -l app=frontend -n trendyol-gift --timeout=300s
```

### 7. Configure Ingress

Update `ingress.yaml` with your domain name:

```yaml
spec:
  tls:
  - hosts:
    - your-domain.com  # Change this
    secretName: trendyol-gift-tls
  rules:
  - host: your-domain.com  # Change this
```

Then apply:

```bash
kubectl apply -f ingress.yaml
```

### 8. Enable Auto-scaling (Optional)

```bash
kubectl apply -f hpa.yaml
```

## Verification

### Check All Resources

```bash
kubectl get all -n trendyol-gift
```

### Check Pod Status

```bash
kubectl get pods -n trendyol-gift
```

### Check Logs

```bash
# Backend logs
kubectl logs -f deployment/backend -n trendyol-gift

# Frontend logs
kubectl logs -f deployment/frontend -n trendyol-gift

# PostgreSQL logs
kubectl logs -f deployment/postgres -n trendyol-gift

# Redis logs
kubectl logs -f deployment/redis -n trendyol-gift
```

### Test Health Endpoints

```bash
# Port forward to test locally
kubectl port-forward -n trendyol-gift svc/backend-service 8000:8000

# In another terminal
curl http://localhost:8000/api/health
```

## Scaling

### Manual Scaling

```bash
# Scale backend
kubectl scale deployment backend --replicas=5 -n trendyol-gift

# Scale frontend
kubectl scale deployment frontend --replicas=5 -n trendyol-gift
```

### Auto-scaling

Auto-scaling is configured via HPA (Horizontal Pod Autoscaler):
- Backend: 3-10 replicas based on CPU (70%) and Memory (80%)
- Frontend: 3-10 replicas based on CPU (70%) and Memory (80%)

## Updating Deployments

### Rolling Update

```bash
# Update backend image
kubectl set image deployment/backend backend=your-registry/trendyol-gift-backend:v2 -n trendyol-gift

# Update frontend image
kubectl set image deployment/frontend frontend=your-registry/trendyol-gift-frontend:v2 -n trendyol-gift

# Check rollout status
kubectl rollout status deployment/backend -n trendyol-gift
kubectl rollout status deployment/frontend -n trendyol-gift
```

### Rollback

```bash
# Rollback backend
kubectl rollout undo deployment/backend -n trendyol-gift

# Rollback frontend
kubectl rollout undo deployment/frontend -n trendyol-gift
```

## Monitoring

### Resource Usage

```bash
kubectl top pods -n trendyol-gift
kubectl top nodes
```

### Events

```bash
kubectl get events -n trendyol-gift --sort-by='.lastTimestamp'
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod to see events
kubectl describe pod <pod-name> -n trendyol-gift

# Check logs
kubectl logs <pod-name> -n trendyol-gift

# Check previous logs if pod restarted
kubectl logs <pod-name> -n trendyol-gift --previous
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/backend -n trendyol-gift -- python -c "
from app.core.config import settings
import psycopg2
conn = psycopg2.connect(settings.DATABASE_URL)
print('Database connection successful!')
"
```

### Redis Connection Issues

```bash
# Test Redis connectivity
kubectl exec -it deployment/redis -n trendyol-gift -- redis-cli -a $REDIS_PASSWORD ping
```

## Backup and Restore

### Database Backup

```bash
# Create backup
kubectl exec -it deployment/postgres -n trendyol-gift -- pg_dump -U postgres trendyol_gift_prod > backup.sql

# Restore backup
kubectl exec -i deployment/postgres -n trendyol-gift -- psql -U postgres trendyol_gift_prod < backup.sql
```

### Redis Backup

```bash
# Create backup
kubectl exec -it deployment/redis -n trendyol-gift -- redis-cli -a $REDIS_PASSWORD SAVE

# Copy backup file
kubectl cp trendyol-gift/redis-pod:/data/dump.rdb ./redis-backup.rdb
```

## Cleanup

### Delete All Resources

```bash
kubectl delete namespace trendyol-gift
```

### Delete Specific Resources

```bash
kubectl delete -f backend-deployment.yaml
kubectl delete -f frontend-deployment.yaml
kubectl delete -f postgres-deployment.yaml
kubectl delete -f redis-deployment.yaml
kubectl delete -f ingress.yaml
kubectl delete -f hpa.yaml
```

## Security Best Practices

1. **Never commit secrets to version control**
2. Use external secret management (AWS Secrets Manager, Vault, etc.)
3. Enable RBAC and limit permissions
4. Use network policies to restrict pod-to-pod communication
5. Enable pod security policies
6. Regularly update images and scan for vulnerabilities
7. Use TLS for all external communication
8. Implement proper backup and disaster recovery procedures

## Production Checklist

- [ ] Secrets are managed securely (not in Git)
- [ ] Domain name is configured in Ingress
- [ ] SSL certificates are configured
- [ ] Resource limits are set appropriately
- [ ] Health checks are configured
- [ ] Monitoring and alerting are set up
- [ ] Backup strategy is implemented
- [ ] Disaster recovery plan is documented
- [ ] Auto-scaling is configured
- [ ] Network policies are in place
- [ ] RBAC is configured
- [ ] Log aggregation is set up
- [ ] Performance testing is completed

## Support

For issues or questions, please contact the development team or refer to the main project documentation.
