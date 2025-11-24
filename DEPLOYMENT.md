# Deployment Guide

This guide covers deployment options for the Trendyol Gift Recommendation application.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Development Deployment](#development-deployment)
- [Production Deployment with Docker Compose](#production-deployment-with-docker-compose)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Post-Deployment](#post-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- Docker 20.10+
- Docker Compose 2.0+
- Node.js 20+ (for local frontend development)
- Python 3.10+ (for local backend development)
- kubectl 1.24+ (for Kubernetes deployment)

### Required Files

- Model checkpoint file: `checkpoints/integrated_enhanced/integrated_enhanced_best.pt`
- Environment configuration files (see [Environment Configuration](#environment-configuration))

## Environment Configuration

The application uses different environment files for different deployment scenarios:

### Development (`.env.development`)

Used for local development with Docker Compose:

```bash
cp .env.example .env.development
# Edit .env.development with your development settings
```

### Staging (`.env.staging`)

Used for staging environment:

```bash
cp .env.example .env.staging
# Edit .env.staging with your staging settings
```

### Production (`.env.production`)

Used for production deployment:

```bash
cp .env.example .env.production
# Edit .env.production with your production settings
```

**⚠️ IMPORTANT**: Never commit `.env.production` with real credentials to version control!

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `REDIS_PASSWORD` | Redis password | Yes |
| `SECRET_KEY` | Application secret key | Yes |
| `TRENDYOL_API_KEY` | Trendyol API key | Yes |
| `TRENDYOL_API_SECRET` | Trendyol API secret | Yes |
| `MODEL_CHECKPOINT_PATH` | Path to model checkpoint | Yes |

## Development Deployment

### Quick Start

```bash
# Run the deployment script
./scripts/deploy-dev.sh
```

### Manual Steps

```bash
# 1. Load environment variables
export $(cat .env.development | grep -v '^#' | xargs)

# 2. Build and start services
docker-compose up -d

# 3. Check service health
docker-compose ps
docker-compose logs -f
```

### Access Services

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Stop Services

```bash
docker-compose down
```

## Production Deployment with Docker Compose

### Prerequisites

1. Ensure all environment variables are set in `.env.production`
2. Verify model checkpoint file exists
3. Configure SSL certificates (if using HTTPS)

### Deployment Steps

```bash
# Run the production deployment script
./scripts/deploy-prod.sh
```

### Manual Steps

```bash
# 1. Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

# 2. Build production images
docker-compose -f docker-compose.prod.yml build

# 3. Start services
docker-compose -f docker-compose.prod.yml up -d

# 4. Check service health
docker-compose -f docker-compose.prod.yml ps
```

### Monitoring

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Check resource usage
docker stats

# Check service health
curl http://localhost:8000/api/health
```

### Backup

```bash
# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U postgres trendyol_gift_prod > backup.sql

# Backup Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD SAVE
docker cp $(docker-compose -f docker-compose.prod.yml ps -q redis):/data/dump.rdb ./redis-backup.rdb
```

## Kubernetes Deployment

### Prerequisites

1. Kubernetes cluster (v1.24+)
2. kubectl configured
3. Docker images built and pushed to registry
4. Persistent storage provisioner
5. Ingress controller (nginx-ingress)
6. cert-manager (optional, for SSL)

### Deployment Steps

```bash
# Run the Kubernetes deployment script
./scripts/deploy-k8s.sh
```

### Manual Steps

See [k8s/README.md](k8s/README.md) for detailed Kubernetes deployment instructions.

### Quick Deploy

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Configure secrets (IMPORTANT!)
kubectl create secret generic trendyol-gift-secrets \
  --namespace=trendyol-gift \
  --from-literal=POSTGRES_PASSWORD=your-password \
  --from-literal=REDIS_PASSWORD=your-password \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=TRENDYOL_API_KEY=your-api-key \
  --from-literal=TRENDYOL_API_SECRET=your-api-secret

# 3. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# 4. Deploy services
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# 5. Deploy Ingress and HPA
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# 6. Check status
kubectl get all -n trendyol-gift
```

## Post-Deployment

### Verification Checklist

- [ ] All services are running
- [ ] Health endpoints return 200 OK
- [ ] Database connection is working
- [ ] Redis connection is working
- [ ] Model loads successfully
- [ ] API endpoints respond correctly
- [ ] Frontend loads and displays correctly
- [ ] SSL certificates are configured (production)
- [ ] Monitoring is set up
- [ ] Backup schedule is configured
- [ ] Logs are being collected

### Health Checks

```bash
# Backend health
curl http://your-domain.com/api/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "trendyol_api_status": "connected",
#   "cache_status": "connected",
#   "timestamp": "2024-01-01T00:00:00Z"
# }
```

### Performance Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://your-domain.com/api/health

# Test with POST request
ab -n 100 -c 5 -p request.json -T application/json http://your-domain.com/api/recommendations
```

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend

# Or for Kubernetes
kubectl logs -f deployment/backend -n trendyol-gift
```

#### Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose exec postgres pg_isready -U postgres

# Check connection string
echo $DATABASE_URL

# Test connection
docker-compose exec backend python -c "
from app.core.config import settings
import psycopg2
conn = psycopg2.connect(settings.DATABASE_URL)
print('Connection successful!')
"
```

#### Redis Connection Failed

```bash
# Check Redis is running
docker-compose exec redis redis-cli ping

# Test with password
docker-compose exec redis redis-cli -a $REDIS_PASSWORD ping
```

#### Model Loading Failed

```bash
# Check model file exists
ls -lh checkpoints/integrated_enhanced/integrated_enhanced_best.pt

# Check file permissions
chmod 644 checkpoints/integrated_enhanced/integrated_enhanced_best.pt

# Check backend logs for detailed error
docker-compose logs backend | grep -i model
```

#### High Memory Usage

```bash
# Check resource usage
docker stats

# For Kubernetes
kubectl top pods -n trendyol-gift

# Adjust resource limits in docker-compose.prod.yml or k8s manifests
```

#### Slow Response Times

```bash
# Check if cache is working
docker-compose exec redis redis-cli -a $REDIS_PASSWORD INFO stats

# Check backend logs for slow queries
docker-compose logs backend | grep -i "slow"

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://your-domain.com/api/health
```

### Getting Help

1. Check application logs
2. Review environment variables
3. Verify all services are running
4. Check network connectivity
5. Review resource usage
6. Consult the main README.md
7. Contact the development team

## Security Considerations

### Production Checklist

- [ ] All default passwords changed
- [ ] Secrets managed securely (not in Git)
- [ ] HTTPS enabled with valid certificates
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Input validation enabled
- [ ] SQL injection protection enabled
- [ ] XSS protection enabled
- [ ] Security headers configured
- [ ] Regular security updates scheduled
- [ ] Backup and disaster recovery tested
- [ ] Monitoring and alerting configured

### Best Practices

1. **Never commit secrets to version control**
2. Use environment variables or secret management tools
3. Enable HTTPS for all production traffic
4. Implement proper authentication and authorization
5. Regular security audits and updates
6. Monitor for suspicious activity
7. Implement proper logging and alerting
8. Regular backups and disaster recovery testing
9. Use least privilege principle for all services
10. Keep all dependencies up to date

## Maintenance

### Regular Tasks

- **Daily**: Check logs and monitoring dashboards
- **Weekly**: Review resource usage and performance metrics
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and test disaster recovery procedures

### Updates

```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Update Kubernetes deployment
kubectl set image deployment/backend backend=new-image:tag -n trendyol-gift
kubectl rollout status deployment/backend -n trendyol-gift
```

### Rollback

```bash
# Docker Compose
docker-compose down
docker-compose up -d --force-recreate

# Kubernetes
kubectl rollout undo deployment/backend -n trendyol-gift
```

## Support

For additional help:
- Review the main [README.md](README.md)
- Check [k8s/README.md](k8s/README.md) for Kubernetes-specific help
- Contact the development team
- Review application logs for detailed error messages
