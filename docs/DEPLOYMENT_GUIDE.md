# Deployment Guide

Bu doküman, Trendyol Gift Recommendation Web uygulamasının farklı ortamlara nasıl deploy edileceğini açıklar.

## İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Development Ortamı](#development-ortamı)
3. [Staging Ortamı](#staging-ortamı)
4. [Production Ortamı](#production-ortamı)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Gereksinimler

### Minimum Sistem Gereksinimleri

**Development:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB
- Docker 20.10+
- Docker Compose 2.0+

**Production:**
- CPU: 8 cores (16 önerilir)
- RAM: 16 GB (32 GB önerilir)
- Disk: 100 GB SSD
- GPU: NVIDIA GPU (4GB+ VRAM) - opsiyonel ama önerilir
- Kubernetes 1.24+

### Yazılım Gereksinimleri

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Nginx 1.20+

## Development Ortamı

### 1. Repository'yi Clone Edin

```bash
git clone https://github.com/your-org/trendyol-gift-recommendation.git
cd trendyol-gift-recommendation
```

### 2. Environment Variables Ayarlayın

```bash
# Backend için
cp backend/.env.example backend/.env

# Frontend için
cp frontend/.env.example frontend/.env

# Root .env dosyası
cp .env.example .env
```

**Backend .env örneği:**
```env
# Application
PROJECT_NAME="Trendyol Gift Recommendation API"
VERSION="1.0.0"
DEBUG=true
ENVIRONMENT=development

# API
API_V1_PREFIX=/api/v1
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/gift_recommendation

# Redis
REDIS_URL=redis://redis:6379/0

# Trendyol API
TRENDYOL_API_KEY=your_api_key_here
TRENDYOL_API_URL=https://api.trendyol.com

# Model
MODEL_CHECKPOINT_PATH=/app/checkpoints/integrated_enhanced/integrated_enhanced_best.pt
DEVICE=cuda  # or cpu

# Security
SECRET_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/app.log
```

**Frontend .env örneği:**
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_TITLE="Trendyol Hediye Önerisi"
VITE_ENABLE_ANALYTICS=false
```

### 3. Docker Compose ile Başlatın

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları takip et
docker-compose logs -f

# Sadece backend
docker-compose up -d backend

# Sadece frontend
docker-compose up -d frontend
```

### 4. Servisleri Kontrol Edin

```bash
# Backend health check
curl http://localhost:8000/api/v1/health

# Frontend
open http://localhost:3000
```

### 5. Development Workflow

```bash
# Backend değişikliklerinde hot reload aktif
# Frontend değişikliklerinde Vite HMR aktif

# Testleri çalıştır
docker-compose exec backend pytest
docker-compose exec frontend npm test

# Linting
docker-compose exec backend black . && mypy .
docker-compose exec frontend npm run lint
```

## Staging Ortamı

### 1. Environment Variables

```bash
cp .env.staging .env
```

**Staging .env:**
```env
ENVIRONMENT=staging
DEBUG=false
DATABASE_URL=postgresql://user:password@staging-db:5432/gift_recommendation
REDIS_URL=redis://staging-redis:6379/0
BACKEND_CORS_ORIGINS=["https://staging.example.com"]
```

### 2. Deploy Script

```bash
# Deploy script'i çalıştır
./scripts/deploy-staging.sh
```

**deploy-staging.sh içeriği:**
```bash
#!/bin/bash
set -e

echo "Deploying to staging..."

# Pull latest code
git pull origin develop

# Build images
docker-compose -f docker-compose.staging.yml build

# Run migrations
docker-compose -f docker-compose.staging.yml run --rm backend alembic upgrade head

# Restart services
docker-compose -f docker-compose.staging.yml up -d

# Run smoke tests
./scripts/smoke-tests.sh staging

echo "Staging deployment complete!"
```

### 3. Smoke Tests

```bash
#!/bin/bash
# smoke-tests.sh

BASE_URL=$1

# Health check
curl -f $BASE_URL/api/v1/health || exit 1

# Test recommendation endpoint
curl -f -X POST $BASE_URL/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_profile": {...}}' || exit 1

echo "Smoke tests passed!"
```

## Production Ortamı

### Option 1: Docker Compose (Küçük Ölçek)

```bash
# Production compose file kullan
docker-compose -f docker-compose.prod.yml up -d

# SSL sertifikalarını ayarla (Let's Encrypt)
./scripts/setup-ssl.sh
```

### Option 2: Kubernetes (Büyük Ölçek)

Aşağıdaki Kubernetes Deployment bölümüne bakın.

## Docker Deployment

### 1. Docker Images Build

```bash
# Backend image
cd backend
docker build -t trendyol-gift-api:latest .

# Frontend image
cd frontend
docker build -t trendyol-gift-web:latest .
```

### 2. Docker Registry'ye Push

```bash
# Tag images
docker tag trendyol-gift-api:latest registry.example.com/trendyol-gift-api:latest
docker tag trendyol-gift-web:latest registry.example.com/trendyol-gift-web:latest

# Push to registry
docker push registry.example.com/trendyol-gift-api:latest
docker push registry.example.com/trendyol-gift-web:latest
```

### 3. Production Docker Compose

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  backend:
    image: registry.example.com/trendyol-gift-api:latest
    restart: always
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  frontend:
    image: registry.example.com/trendyol-gift-web:latest
    restart: always
    deploy:
      replicas: 2

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend

  postgres:
    image: postgres:14-alpine
    restart: always
    environment:
      POSTGRES_DB: gift_recommendation
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment

### 1. Namespace Oluştur

```bash
kubectl apply -f k8s/namespace.yaml
```

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gift-recommendation
```

### 2. Secrets Oluştur

```bash
# Database credentials
kubectl create secret generic db-credentials \
  --from-literal=username=dbuser \
  --from-literal=password=dbpassword \
  -n gift-recommendation

# API keys
kubectl create secret generic api-keys \
  --from-literal=trendyol-api-key=your-key \
  --from-literal=secret-key=your-secret \
  -n gift-recommendation
```

### 3. ConfigMap Oluştur

```bash
kubectl apply -f k8s/configmap.yaml
```

### 4. Deploy Services

```bash
# PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml

# Redis
kubectl apply -f k8s/redis-deployment.yaml

# Backend
kubectl apply -f k8s/backend-deployment.yaml

# Frontend
kubectl apply -f k8s/frontend-deployment.yaml

# Ingress
kubectl apply -f k8s/ingress.yaml
```

### 5. Horizontal Pod Autoscaler

```bash
kubectl apply -f k8s/hpa.yaml
```

**hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: gift-recommendation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 6. Deployment Doğrulama

```bash
# Pod'ları kontrol et
kubectl get pods -n gift-recommendation

# Service'leri kontrol et
kubectl get svc -n gift-recommendation

# Logs
kubectl logs -f deployment/backend -n gift-recommendation

# Health check
kubectl exec -it deployment/backend -n gift-recommendation -- \
  curl http://localhost:8000/api/v1/health
```

## Monitoring Setup

### 1. Prometheus & Grafana

```bash
# Monitoring stack'i deploy et
docker-compose -f docker-compose.monitoring.yml up -d

# veya Kubernetes için
kubectl apply -f k8s/monitoring/
```

### 2. Grafana Dashboards

Grafana'ya erişin: http://localhost:3001

**Default credentials:**
- Username: admin
- Password: admin

**Import dashboards:**
1. API Performance: `monitoring/grafana/dashboards/api-performance.json`
2. System Resources: `monitoring/grafana/dashboards/system-resources.json`
3. Tool Analytics: `monitoring/grafana/dashboards/tool-analytics.json`

### 3. Alerting

**Alertmanager configuration:**
```yaml
# monitoring/alertmanager/alertmanager.yml
route:
  receiver: 'email'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
- name: 'email'
  email_configs:
  - to: 'ops@example.com'
    from: 'alertmanager@example.com'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alertmanager@example.com'
    auth_password: 'password'
```

## SSL/TLS Setup

### Let's Encrypt ile SSL

```bash
# Certbot kurulumu
sudo apt-get install certbot python3-certbot-nginx

# Sertifika al
sudo certbot --nginx -d example.com -d www.example.com

# Auto-renewal test
sudo certbot renew --dry-run
```

### Manual SSL Certificate

```bash
# Self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key \
  -out ssl/certificate.crt
```

## Database Migrations

### Alembic Migrations

```bash
# Yeni migration oluştur
docker-compose exec backend alembic revision --autogenerate -m "description"

# Migration'ları uygula
docker-compose exec backend alembic upgrade head

# Rollback
docker-compose exec backend alembic downgrade -1
```

## Backup & Recovery

### Database Backup

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
docker-compose exec -T postgres pg_dump -U user gift_recommendation | \
  gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Redis backup
docker-compose exec redis redis-cli SAVE
docker cp gift-recommendation_redis_1:/data/dump.rdb \
  $BACKUP_DIR/redis_backup_$DATE.rdb
```

### Restore

```bash
# PostgreSQL restore
gunzip < backup.sql.gz | \
  docker-compose exec -T postgres psql -U user gift_recommendation

# Redis restore
docker cp backup.rdb gift-recommendation_redis_1:/data/dump.rdb
docker-compose restart redis
```

## Troubleshooting

### Common Issues

**1. Model yüklenmiyor**
```bash
# Checkpoint dosyasını kontrol et
ls -lh checkpoints/integrated_enhanced/

# Backend logs
docker-compose logs backend | grep -i "model"

# Manuel test
docker-compose exec backend python -c "import torch; print(torch.cuda.is_available())"
```

**2. Trendyol API bağlantı hatası**
```bash
# API key kontrolü
docker-compose exec backend env | grep TRENDYOL

# Network testi
docker-compose exec backend curl -v https://api.trendyol.com
```

**3. Redis bağlantı hatası**
```bash
# Redis durumu
docker-compose exec redis redis-cli ping

# Connection test
docker-compose exec backend python -c "import redis; r=redis.from_url('redis://redis:6379'); print(r.ping())"
```

**4. High memory usage**
```bash
# Container stats
docker stats

# Memory profiling
docker-compose exec backend python -m memory_profiler app/main.py
```

### Log Analysis

```bash
# Backend errors
docker-compose logs backend | grep ERROR

# Slow requests
docker-compose logs backend | grep "inference_time" | awk '{if($NF>2.0) print}'

# Rate limit violations
docker-compose logs nginx | grep "429"
```

### Performance Tuning

**Backend:**
```python
# Gunicorn workers
workers = (cpu_count * 2) + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
```

**PostgreSQL:**
```sql
-- Connection pooling
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
```

**Redis:**
```conf
maxmemory 512mb
maxmemory-policy allkeys-lru
```

## CI/CD Pipeline

### GitHub Actions

**.github/workflows/deploy.yml:**
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build images
        run: |
          docker build -t backend:${{ github.sha }} backend/
          docker build -t frontend:${{ github.sha }} frontend/
      
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
      
      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
          docker push backend:${{ github.sha }}
          docker push frontend:${{ github.sha }}
      
      - name: Deploy to production
        run: |
          ./scripts/deploy-prod.sh ${{ github.sha }}
```

## Rollback Procedure

```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend -n gift-recommendation

# Docker Compose rollback
docker-compose pull  # Pull previous version
docker-compose up -d

# Database rollback
docker-compose exec backend alembic downgrade -1
```

## Support & Maintenance

- **Monitoring Dashboard:** https://grafana.example.com
- **Logs:** https://kibana.example.com
- **Status Page:** https://status.example.com
- **On-call:** ops@example.com
