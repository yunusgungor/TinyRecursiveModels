# Production Deployment Kılavuzu

Bu doküman, uygulamanın production ortamına deploy edilmesi için gerekli adımları ve en iyi uygulamaları açıklar.

## İçindekiler

- [Ön Hazırlık](#ön-hazırlık)
- [Build ve Push Süreci](#build-ve-push-süreci)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring ve Troubleshooting](#monitoring-ve-troubleshooting)
- [Rollback Prosedürü](#rollback-prosedürü)
- [Güvenlik Kontrolleri](#güvenlik-kontrolleri)

## Ön Hazırlık

### Gereksinimler

- **Docker**: 20.10+ (BuildKit desteği ile)
- **kubectl**: Kubernetes cluster'a erişim için
- **Image Registry**: Docker Hub, GitHub Container Registry, veya özel registry
- **Kubernetes Cluster**: 1.24+ versiyonu
- **Helm** (opsiyonel): Chart yönetimi için

### Environment Değişkenleri

Production deployment için gerekli environment değişkenlerini ayarlayın:

```bash
# Registry bilgileri
export REGISTRY=ghcr.io/your-org
export VERSION=$(git rev-parse --short HEAD)

# Kubernetes context
export KUBE_CONTEXT=production

# Secrets
export DB_PASSWORD=<secure-password>
export REDIS_PASSWORD=<secure-password>
export SECRET_KEY=<secure-secret-key>
```

### Pre-deployment Checklist

- [ ] Tüm testler geçiyor mu?
- [ ] Code review tamamlandı mı?
- [ ] Security scan yapıldı mı?
- [ ] Database migration'lar hazır mı?
- [ ] Monitoring ve alerting yapılandırıldı mı?
- [ ] Rollback planı hazır mı?
- [ ] Stakeholder'lar bilgilendirildi mi?

## Build ve Push Süreci

### 1. Production Image Build

#### BuildKit ile Optimize Edilmiş Build

```bash
# BuildKit'i etkinleştir
export DOCKER_BUILDKIT=1

# Backend image build
docker build \
  --target production \
  --cache-from ${REGISTRY}/backend:cache \
  --cache-to type=inline \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --tag ${REGISTRY}/backend:${VERSION} \
  --tag ${REGISTRY}/backend:latest \
  backend/

# Frontend image build
docker build \
  --target production \
  --cache-from ${REGISTRY}/frontend:cache \
  --cache-to type=inline \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --tag ${REGISTRY}/frontend:${VERSION} \
  --tag ${REGISTRY}/frontend:latest \
  frontend/
```

#### Docker Compose ile Build

```bash
# docker-compose.prod.yml kullanarak build
docker-compose -f docker-compose.prod.yml build

# Version tag'leri ekle
docker tag myapp_backend:latest ${REGISTRY}/backend:${VERSION}
docker tag myapp_frontend:latest ${REGISTRY}/frontend:${VERSION}
```

### 2. Image Güvenlik Taraması

Build'den sonra güvenlik açıklarını tarayın:

```bash
# Trivy ile tarama
trivy image ${REGISTRY}/backend:${VERSION}
trivy image ${REGISTRY}/frontend:${VERSION}

# Kritik ve yüksek seviye açıklar varsa build'i durdur
trivy image --severity HIGH,CRITICAL --exit-code 1 ${REGISTRY}/backend:${VERSION}
```

### 3. Image'ları Registry'ye Push

```bash
# Registry'ye login
docker login ${REGISTRY}

# Backend push
docker push ${REGISTRY}/backend:${VERSION}
docker push ${REGISTRY}/backend:latest

# Frontend push
docker push ${REGISTRY}/frontend:${VERSION}
docker push ${REGISTRY}/frontend:latest}

# Cache layer'larını push et
docker push ${REGISTRY}/backend:cache
docker push ${REGISTRY}/frontend:cache
```

### 4. Image Doğrulama

```bash
# Image'ın registry'de olduğunu doğrula
docker pull ${REGISTRY}/backend:${VERSION}
docker pull ${REGISTRY}/frontend:${VERSION}

# Image boyutlarını kontrol et
docker images | grep ${VERSION}

# Beklenen boyutlar:
# Backend: < 200MB
# Frontend: < 50MB
```

## Kubernetes Deployment

### 1. Cluster Bağlantısı

```bash
# Kubernetes context'i ayarla
kubectl config use-context ${KUBE_CONTEXT}

# Cluster bağlantısını doğrula
kubectl cluster-info
kubectl get nodes
```

### 2. Namespace Oluşturma

```bash
# Namespace oluştur (ilk deployment için)
kubectl apply -f k8s/namespace.yaml

# Namespace'i doğrula
kubectl get namespace production
```

### 3. Secrets ve ConfigMaps

```bash
# Secrets oluştur
kubectl create secret generic app-secrets \
  --from-literal=db-password=${DB_PASSWORD} \
  --from-literal=redis-password=${REDIS_PASSWORD} \
  --from-literal=secret-key=${SECRET_KEY} \
  --namespace=production \
  --dry-run=client -o yaml | kubectl apply -f -

# ConfigMap oluştur
kubectl apply -f k8s/configmap.yaml --namespace=production

# Secrets ve ConfigMaps'i doğrula
kubectl get secrets --namespace=production
kubectl get configmaps --namespace=production
```

### 4. Database Deployment

```bash
# PostgreSQL StatefulSet deploy et
kubectl apply -f k8s/postgres-deployment.yaml --namespace=production

# Redis StatefulSet deploy et
kubectl apply -f k8s/redis-deployment.yaml --namespace=production

# Pod'ların hazır olmasını bekle
kubectl wait --for=condition=ready pod -l app=postgres --namespace=production --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --namespace=production --timeout=300s
```

### 5. Application Deployment

```bash
# Backend deployment
kubectl apply -f k8s/backend-deployment.yaml --namespace=production

# Frontend deployment
kubectl apply -f k8s/frontend-deployment.yaml --namespace=production

# Deployment durumunu izle
kubectl rollout status deployment/backend --namespace=production
kubectl rollout status deployment/frontend --namespace=production
```

### 6. Service ve Ingress

```bash
# Services oluştur
kubectl apply -f k8s/backend-service.yaml --namespace=production
kubectl apply -f k8s/frontend-service.yaml --namespace=production

# Ingress oluştur
kubectl apply -f k8s/ingress.yaml --namespace=production

# Ingress IP'sini al
kubectl get ingress --namespace=production
```

### 7. HorizontalPodAutoscaler

```bash
# HPA oluştur
kubectl apply -f k8s/hpa.yaml --namespace=production

# HPA durumunu kontrol et
kubectl get hpa --namespace=production
```

### 8. Deployment Doğrulama

```bash
# Tüm pod'ların çalıştığını doğrula
kubectl get pods --namespace=production

# Beklenen çıktı:
# NAME                        READY   STATUS    RESTARTS   AGE
# backend-xxx-yyy            1/1     Running   0          2m
# backend-xxx-zzz            1/1     Running   0          2m
# frontend-xxx-yyy           1/1     Running   0          2m
# postgres-0                 1/1     Running   0          5m
# redis-0                    1/1     Running   0          5m

# Service endpoint'lerini test et
kubectl run curl-test --image=curlimages/curl -i --rm --restart=Never -- \
  curl http://backend-service:8000/health

# Ingress üzerinden erişimi test et
curl https://your-domain.com/health
```

## CI/CD Pipeline Entegrasyonu

### GitHub Actions Örneği

```yaml
name: Production Deployment

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push Backend
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          target: production
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:${{ github.sha }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:cache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:cache,mode=max

      - name: Build and Push Frontend
        uses: docker/build-push-action@v4
        with:
          context: ./frontend
          target: production
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:cache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:cache,mode=max

      - name: Security Scan
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image --severity HIGH,CRITICAL \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:${{ github.sha }}

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/backend \
            backend=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:${{ github.sha }} \
            --namespace=production
          
          kubectl set image deployment/frontend \
            frontend=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }} \
            --namespace=production

      - name: Verify Deployment
        run: |
          kubectl rollout status deployment/backend --namespace=production
          kubectl rollout status deployment/frontend --namespace=production
```

### Deployment Script

Alternatif olarak, deployment script kullanabilirsiniz:

```bash
#!/bin/bash
# scripts/deploy-prod.sh

set -e

VERSION=${1:-$(git rev-parse --short HEAD)}
REGISTRY=${REGISTRY:-ghcr.io/your-org}
NAMESPACE=${NAMESPACE:-production}

echo "🚀 Starting production deployment..."
echo "Version: ${VERSION}"
echo "Registry: ${REGISTRY}"
echo "Namespace: ${NAMESPACE}"

# Build images
echo "📦 Building images..."
docker-compose -f docker-compose.prod.yml build

# Tag images
echo "🏷️  Tagging images..."
docker tag myapp_backend:latest ${REGISTRY}/backend:${VERSION}
docker tag myapp_frontend:latest ${REGISTRY}/frontend:${VERSION}

# Security scan
echo "🔒 Running security scan..."
trivy image --severity HIGH,CRITICAL --exit-code 1 ${REGISTRY}/backend:${VERSION}
trivy image --severity HIGH,CRITICAL --exit-code 1 ${REGISTRY}/frontend:${VERSION}

# Push images
echo "⬆️  Pushing images..."
docker push ${REGISTRY}/backend:${VERSION}
docker push ${REGISTRY}/frontend:${VERSION}

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl set image deployment/backend backend=${REGISTRY}/backend:${VERSION} -n ${NAMESPACE}
kubectl set image deployment/frontend frontend=${REGISTRY}/frontend:${VERSION} -n ${NAMESPACE}

# Wait for rollout
echo "⏳ Waiting for rollout..."
kubectl rollout status deployment/backend -n ${NAMESPACE}
kubectl rollout status deployment/frontend -n ${NAMESPACE}

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n ${NAMESPACE}

echo "🎉 Deployment completed successfully!"
```

Kullanım:
```bash
chmod +x scripts/deploy-prod.sh
./scripts/deploy-prod.sh v1.2.3
```

## Monitoring ve Troubleshooting

### Pod Durumunu İzleme

```bash
# Tüm pod'ları listele
kubectl get pods --namespace=production

# Pod detaylarını görüntüle
kubectl describe pod <pod-name> --namespace=production

# Pod loglarını görüntüle
kubectl logs <pod-name> --namespace=production

# Canlı log takibi
kubectl logs -f <pod-name> --namespace=production

# Önceki container'ın logları (crash durumunda)
kubectl logs <pod-name> --previous --namespace=production
```

### Resource Kullanımı

```bash
# Pod resource kullanımı
kubectl top pods --namespace=production

# Node resource kullanımı
kubectl top nodes

# HPA durumu
kubectl get hpa --namespace=production
```

### Health Check'ler

```bash
# Backend health check
kubectl exec -it <backend-pod> --namespace=production -- \
  curl http://localhost:8000/health

# Frontend health check
kubectl exec -it <frontend-pod> --namespace=production -- \
  curl http://localhost:80/

# Database bağlantısı
kubectl exec -it postgres-0 --namespace=production -- \
  psql -U postgres -c "SELECT 1"
```

### Events İzleme

```bash
# Namespace event'lerini görüntüle
kubectl get events --namespace=production --sort-by='.lastTimestamp'

# Belirli bir pod'un event'leri
kubectl get events --namespace=production --field-selector involvedObject.name=<pod-name>
```

### Prometheus Metrics

```bash
# Metrics endpoint'i kontrol et
kubectl port-forward svc/backend-service 8000:8000 --namespace=production
curl http://localhost:8000/metrics
```

## Rollback Prosedürü

### Hızlı Rollback

```bash
# Son deployment'ı geri al
kubectl rollout undo deployment/backend --namespace=production
kubectl rollout undo deployment/frontend --namespace=production

# Belirli bir revision'a geri dön
kubectl rollout undo deployment/backend --to-revision=2 --namespace=production
```

### Rollback Script

```bash
#!/bin/bash
# scripts/rollback.sh

set -e

NAMESPACE=${NAMESPACE:-production}
REVISION=${1}

echo "🔄 Starting rollback..."

if [ -z "$REVISION" ]; then
  echo "Rolling back to previous version..."
  kubectl rollout undo deployment/backend -n ${NAMESPACE}
  kubectl rollout undo deployment/frontend -n ${NAMESPACE}
else
  echo "Rolling back to revision ${REVISION}..."
  kubectl rollout undo deployment/backend --to-revision=${REVISION} -n ${NAMESPACE}
  kubectl rollout undo deployment/frontend --to-revision=${REVISION} -n ${NAMESPACE}
fi

echo "⏳ Waiting for rollback..."
kubectl rollout status deployment/backend -n ${NAMESPACE}
kubectl rollout status deployment/frontend -n ${NAMESPACE}

echo "✅ Rollback completed!"
kubectl get pods -n ${NAMESPACE}
```

### Deployment History

```bash
# Deployment geçmişini görüntüle
kubectl rollout history deployment/backend --namespace=production

# Belirli bir revision'ın detaylarını gör
kubectl rollout history deployment/backend --revision=3 --namespace=production
```

## Güvenlik Kontrolleri

### Image Güvenliği

```bash
# Trivy ile tarama
trivy image ${REGISTRY}/backend:${VERSION}

# Sadece kritik açıkları göster
trivy image --severity CRITICAL ${REGISTRY}/backend:${VERSION}

# JSON formatında rapor
trivy image --format json --output report.json ${REGISTRY}/backend:${VERSION}
```

### Secret Yönetimi

```bash
# Secret'ları listele (değerleri göstermez)
kubectl get secrets --namespace=production

# Secret'ı güncelle
kubectl create secret generic app-secrets \
  --from-literal=db-password=${NEW_DB_PASSWORD} \
  --namespace=production \
  --dry-run=client -o yaml | kubectl apply -f -

# Secret değişikliğinden sonra pod'ları restart et
kubectl rollout restart deployment/backend --namespace=production
```

### Network Policies

```bash
# Network policy uygula
kubectl apply -f k8s/network-policy.yaml --namespace=production

# Network policy'leri listele
kubectl get networkpolicies --namespace=production
```

### RBAC Kontrolleri

```bash
# Service account oluştur
kubectl create serviceaccount app-sa --namespace=production

# Role binding oluştur
kubectl create rolebinding app-binding \
  --role=app-role \
  --serviceaccount=production:app-sa \
  --namespace=production
```

## Performance Optimization

### Resource Limits

```yaml
# k8s/backend-deployment.yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Autoscaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Cache Optimization

```bash
# Image pull policy'yi optimize et
imagePullPolicy: IfNotPresent

# Registry cache kullan
--cache-from type=registry,ref=${REGISTRY}/backend:cache
```

## Backup ve Disaster Recovery

### Database Backup

```bash
# PostgreSQL backup
kubectl exec postgres-0 --namespace=production -- \
  pg_dump -U postgres mydb > backup-$(date +%Y%m%d).sql

# Backup'ı restore et
kubectl exec -i postgres-0 --namespace=production -- \
  psql -U postgres mydb < backup-20231201.sql
```

### Persistent Volume Backup

```bash
# PV'leri listele
kubectl get pv

# PVC'leri listele
kubectl get pvc --namespace=production

# Volume snapshot oluştur (CSI driver gerekli)
kubectl apply -f k8s/volume-snapshot.yaml
```

## Sonraki Adımlar

- [Monitoring Setup](./MONITORING.md) - Prometheus ve Grafana kurulumu
- [Sorun Giderme](./TROUBLESHOOTING.md) - Production sorunları ve çözümleri
- [Scaling Guide](./SCALING.md) - Horizontal ve vertical scaling
