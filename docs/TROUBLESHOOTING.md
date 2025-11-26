# Sorun Giderme Kılavuzu

Bu doküman, container altyapısında karşılaşılabilecek yaygın sorunları ve çözümlerini içerir.

## İçindekiler

- [Build Sorunları](#build-sorunları)
- [Cache Sorunları](#cache-sorunları)
- [Runtime Sorunları](#runtime-sorunları)
- [Network Sorunları](#network-sorunları)
- [Performance Sorunları](#performance-sorunları)
- [Kubernetes Sorunları](#kubernetes-sorunları)
- [Debug Araçları](#debug-araçları)

## Build Sorunları

### Sorun: BuildKit Etkin Değil

**Belirtiler**:
```
DEPRECATED: The legacy builder is deprecated and will be removed in a future release.
```

**Çözüm**:
```bash
# BuildKit'i etkinleştir
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Kalıcı yapmak için ~/.bashrc veya ~/.zshrc'ye ekle
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
source ~/.bashrc

# Docker daemon'da etkinleştir (opsiyonel)
# /etc/docker/daemon.json
{
  "features": {
    "buildkit": true
  }
}

# Docker'ı restart et
sudo systemctl restart docker
```

### Sorun: Build Çok Yavaş

**Belirtiler**:
- Build 10 dakikadan uzun sürüyor
- Her build'de tüm layer'lar yeniden build ediliyor

**Çözüm 1: Cache'i kontrol et**
```bash
# BuildKit cache mount kullanıldığını doğrula
grep "mount=type=cache" backend/Dockerfile

# Beklenen:
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install -r requirements.txt
```

**Çözüm 2: .dockerignore'u optimize et**
```bash
# .dockerignore dosyasını kontrol et
cat backend/.dockerignore

# Gereksiz dosyaların hariç tutulduğundan emin ol
**/__pycache__
**/*.pyc
**/node_modules
**/.git
**/logs
```

**Çözüm 3: Layer sırasını optimize et**
```dockerfile
# YANLIŞ - Kod değişikliği tüm layer'ları invalidate eder
COPY . .
RUN pip install -r requirements.txt

# DOĞRU - Dependency layer cache'lenir
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

**Çözüm 4: Paralel build kullan**
```bash
# BuildKit paralel build yapar
docker-compose build --parallel
```

### Sorun: "No Space Left on Device"

**Belirtiler**:
```
ERROR: failed to solve: write /var/lib/docker/...: no space left on device
```

**Çözüm**:
```bash
# Disk kullanımını kontrol et
df -h
docker system df

# Kullanılmayan image'ları temizle
docker image prune -a

# Kullanılmayan container'ları temizle
docker container prune

# Kullanılmayan volume'ları temizle
docker volume prune

# Tüm kullanılmayan kaynakları temizle (DİKKAT: Tüm cache silinir)
docker system prune -a --volumes

# Build cache'i temizle
docker builder prune -a
```

### Sorun: Dependency Installation Başarısız

**Belirtiler**:
```
ERROR: Could not find a version that satisfies the requirement
npm ERR! 404 Not Found
```

**Çözüm 1: Network bağlantısını kontrol et**
```bash
# Container içinden internet erişimini test et
docker run --rm alpine ping -c 3 google.com

# DNS çözümlemesini test et
docker run --rm alpine nslookup pypi.org
```

**Çözüm 2: Registry mirror kullan**
```dockerfile
# Python için
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# npm için
RUN npm config set registry https://registry.npmmirror.com
RUN npm install
```

**Çözüm 3: Proxy ayarlarını yapılandır**
```dockerfile
# Dockerfile'da
ENV HTTP_PROXY=http://proxy.example.com:8080
ENV HTTPS_PROXY=http://proxy.example.com:8080
```

### Sorun: Multi-stage Build Hatası

**Belirtiler**:
```
ERROR: failed to solve: failed to compute cache key: "/app/dist" not found
```

**Çözüm**:
```dockerfile
# Stage isimlerinin doğru olduğundan emin ol
FROM node:20-alpine as builder
RUN npm run build

FROM nginx:alpine as production
# "builder" stage'inden kopyala
COPY --from=builder /app/dist /usr/share/nginx/html
```

## Cache Sorunları

### Sorun: Cache Hiç Kullanılmıyor

**Belirtiler**:
- Her build'de tüm layer'lar yeniden build ediliyor
- Build süreleri hiç azalmıyor

**Çözüm 1: BuildKit cache mount'u kontrol et**
```bash
# Dockerfile'da cache mount olduğunu doğrula
grep "mount=type=cache" backend/Dockerfile

# Yoksa ekle
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

**Çözüm 2: Layer invalidation'ı kontrol et**
```bash
# requirements.txt değişmeden build et
docker-compose build backend

# Log'da "Using cache" mesajını ara
# Görmüyorsan, layer sıralamasını kontrol et
```

**Çözüm 3: Registry cache'i yapılandır**
```bash
# docker-compose.yml'de cache_from ekle
services:
  backend:
    build:
      cache_from:
        - ${REGISTRY}/backend:cache
      args:
        BUILDKIT_INLINE_CACHE: 1
```

### Sorun: Cache Bozuk

**Belirtiler**:
- Build başarılı ama uygulama çalışmıyor
- Eski kod çalışıyor gibi görünüyor

**Çözüm**:
```bash
# Cache'i temizle ve yeniden build et
docker-compose build --no-cache backend

# Veya tüm build cache'i temizle
docker builder prune -a

# Container'ları durdur ve volume'ları temizle
docker-compose down -v

# Yeniden başlat
docker-compose up --build
```

### Sorun: CI/CD'de Cache Çalışmıyor

**Belirtiler**:
- Local'de cache çalışıyor ama CI'da çalışmıyor
- Her CI build'i sıfırdan başlıyor

**Çözüm 1: Registry cache yapılandır**
```yaml
# .github/workflows/build.yml
- name: Build with cache
  uses: docker/build-push-action@v4
  with:
    cache-from: type=registry,ref=${{ env.REGISTRY }}/backend:cache
    cache-to: type=registry,ref=${{ env.REGISTRY }}/backend:cache,mode=max
```

**Çözüm 2: GitHub Actions cache kullan**
```yaml
- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ hashFiles('**/requirements.txt') }}
```

## Runtime Sorunları

### Sorun: Container Başlamıyor

**Belirtiler**:
```
docker-compose ps
backend_1   Exit 1
```

**Çözüm 1: Logları kontrol et**
```bash
# Container loglarını görüntüle
docker-compose logs backend

# Son 100 satırı göster
docker-compose logs --tail=100 backend

# Canlı takip
docker-compose logs -f backend
```

**Çözüm 2: Container'ı manuel başlat**
```bash
# Interactive modda başlat
docker-compose run --rm backend bash

# Komutları manuel çalıştır
python -m app.main
```

**Çözüm 3: Environment değişkenlerini kontrol et**
```bash
# Container içindeki env'leri görüntüle
docker-compose exec backend env

# Eksik değişken varsa .env dosyasına ekle
echo "MISSING_VAR=value" >> backend/.env
```

### Sorun: Health Check Başarısız

**Belirtiler**:
```
docker-compose ps
backend_1   Up (unhealthy)
```

**Çözüm 1: Health endpoint'i test et**
```bash
# Container içinden test et
docker-compose exec backend curl http://localhost:8000/health

# Host'tan test et
curl http://localhost:8000/health

# Beklenen: {"status": "healthy"}
```

**Çözüm 2: Health check timing'i ayarla**
```yaml
# docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s      # Daha uzun interval
  timeout: 5s        # Daha uzun timeout
  retries: 5         # Daha fazla retry
  start_period: 30s  # Başlangıç için daha fazla süre
```

**Çözüm 3: Bağımlılıkları kontrol et**
```bash
# Database bağlantısını test et
docker-compose exec backend python -c "
from app.core.database import engine
engine.connect()
print('Database connection OK')
"

# Redis bağlantısını test et
docker-compose exec backend python -c "
import redis
r = redis.Redis(host='redis')
r.ping()
print('Redis connection OK')
"
```

### Sorun: Hot Reload Çalışmıyor

**Belirtiler**:
- Kod değişiklikleri container'a yansımıyor
- Manuel restart gerekiyor

**Çözüm 1: Volume mount'u kontrol et**
```bash
# docker-compose.yml'de volume mount olduğunu doğrula
docker-compose config | grep -A 5 volumes

# Beklenen:
# volumes:
#   - ./backend:/app
```

**Çözüm 2: File watcher'ı kontrol et**
```bash
# Backend için uvicorn reload aktif mi?
docker-compose exec backend ps aux | grep uvicorn

# Beklenen: --reload flag'i olmalı
# uvicorn app.main:app --reload --host 0.0.0.0

# Frontend için Vite HMR aktif mi?
docker-compose logs frontend | grep HMR
```

**Çözüm 3: File permissions kontrol et**
```bash
# Host'ta dosya sahipliğini kontrol et
ls -la backend/app/

# Container içinde kontrol et
docker-compose exec backend ls -la /app/

# Gerekirse sahipliği düzelt
sudo chown -R $USER:$USER backend/
```

### Sorun: Container Sürekli Restart Oluyor

**Belirtiler**:
```
docker-compose ps
backend_1   Restarting
```

**Çözüm 1: Restart policy'yi kontrol et**
```yaml
# docker-compose.yml
services:
  backend:
    restart: unless-stopped  # "always" yerine
```

**Çözüm 2: OOM (Out of Memory) kontrol et**
```bash
# Container event'lerini kontrol et
docker events --filter 'event=oom'

# Memory kullanımını izle
docker stats backend_1

# Memory limit'i artır
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G  # 2G'den artır
```

**Çözüm 3: Application crash'ini debug et**
```bash
# Crash öncesi logları görüntüle
docker-compose logs --tail=200 backend

# Core dump varsa analiz et
docker-compose exec backend ls -la /tmp/core*
```

## Network Sorunları

### Sorun: Servisler Birbirine Erişemiyor

**Belirtiler**:
```
Connection refused
Could not connect to database
```

**Çözüm 1: Service name'leri kontrol et**
```bash
# docker-compose.yml'de service name'leri doğrula
docker-compose config | grep -A 2 services

# Backend'den database'e bağlan
docker-compose exec backend ping postgres

# Service discovery çalışıyor mu?
docker-compose exec backend nslookup postgres
```

**Çözüm 2: Network'ü kontrol et**
```bash
# Network'leri listele
docker network ls

# Network detaylarını görüntüle
docker network inspect <project>_default

# Container'ların aynı network'te olduğunu doğrula
docker-compose ps -q | xargs docker inspect --format='{{.Name}} - {{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}'
```

**Çözüm 3: Port'ları kontrol et**
```bash
# Container içindeki port'ları kontrol et
docker-compose exec backend netstat -tlnp

# Service'in doğru port'ta dinlediğini doğrula
docker-compose exec backend curl http://localhost:8000/health
```

### Sorun: Port Zaten Kullanımda

**Belirtiler**:
```
ERROR: for backend  Cannot start service backend: 
Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Çözüm 1: Portu kullanan işlemi bul**
```bash
# Linux/Mac
lsof -i :8000
sudo netstat -tlnp | grep 8000

# İşlemi sonlandır
kill -9 <PID>
```

**Çözüm 2: Farklı port kullan**
```yaml
# docker-compose.yml
services:
  backend:
    ports:
      - "8001:8000"  # 8000 yerine 8001 kullan
```

**Çözüm 3: Eski container'ları temizle**
```bash
# Durdurulmuş container'ları temizle
docker-compose down

# Tüm durdurulmuş container'ları sil
docker container prune
```

## Performance Sorunları

### Sorun: Build Çok Yavaş (>10 dakika)

**Tanı**:
```bash
# Build süresini ölç
time docker-compose build backend

# Layer'ları analiz et
docker history ${REGISTRY}/backend:latest
```

**Çözüm 1: Layer cache'i optimize et**
```dockerfile
# Sık değişen dosyaları sona koy
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # En sona
```

**Çözüm 2: .dockerignore'u optimize et**
```bash
# Büyük dosyaları hariç tut
echo "**/*.log" >> .dockerignore
echo "**/node_modules" >> .dockerignore
echo "**/__pycache__" >> .dockerignore
echo "**/dist" >> .dockerignore
```

**Çözüm 3: Multi-stage build kullan**
```dockerfile
# Build dependencies'i ayrı stage'de
FROM python:3.10-slim as builder
RUN pip install -r requirements.txt --target=/packages

# Production'da sadece packages'i kopyala
FROM python:3.10-slim as production
COPY --from=builder /packages /packages
```

### Sorun: Container Yavaş Çalışıyor

**Tanı**:
```bash
# Resource kullanımını izle
docker stats

# CPU ve memory kullanımını kontrol et
docker stats --no-stream backend_1
```

**Çözüm 1: Resource limit'leri artır**
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'      # 1'den artır
          memory: 4G     # 2G'den artır
        reservations:
          cpus: '1'
          memory: 2G
```

**Çözüm 2: Volume mount performansını optimize et**
```yaml
# Mac/Windows için delegated mode kullan
volumes:
  - ./backend:/app:delegated

# Veya named volume kullan
volumes:
  - backend-code:/app
```

**Çözüm 3: Application profiling**
```bash
# Python profiling
docker-compose exec backend python -m cProfile -o profile.stats app/main.py

# Memory profiling
docker-compose exec backend python -m memory_profiler app/main.py
```

### Sorun: Hot Reload Yavaş (>5 saniye)

**Tanı**:
```bash
# File watcher event'lerini izle
docker-compose logs -f backend | grep -i reload

# Değişiklik zamanını ölç
time touch backend/app/main.py
```

**Çözüm 1: Polling interval'i optimize et**
```python
# Backend için uvicorn config
uvicorn.run(
    "app.main:app",
    reload=True,
    reload_delay=0.5  # Default 0.25
)
```

**Çözüm 2: Exclude pattern'leri ekle**
```python
# Gereksiz dosyaları hariç tut
uvicorn.run(
    "app.main:app",
    reload=True,
    reload_excludes=["*.log", "*.pyc", "__pycache__"]
)
```

**Çözüm 3: File system event'lerini optimize et**
```yaml
# docker-compose.yml - Mac için
volumes:
  - ./backend:/app:cached  # delegated yerine cached
```

## Kubernetes Sorunları

### Sorun: Pod Başlamıyor

**Tanı**:
```bash
# Pod durumunu kontrol et
kubectl get pods -n production

# Pod detaylarını görüntüle
kubectl describe pod <pod-name> -n production

# Event'leri kontrol et
kubectl get events -n production --sort-by='.lastTimestamp'
```

**Çözüm 1: Image pull sorunları**
```bash
# Image pull policy'yi kontrol et
kubectl get pod <pod-name> -n production -o yaml | grep imagePullPolicy

# Image'ın registry'de olduğunu doğrula
docker pull ${REGISTRY}/backend:${VERSION}

# Image pull secret'ı kontrol et
kubectl get secrets -n production
kubectl describe secret regcred -n production
```

**Çözüm 2: Resource yetersizliği**
```bash
# Node resource'larını kontrol et
kubectl top nodes

# Pod resource request'lerini kontrol et
kubectl get pod <pod-name> -n production -o yaml | grep -A 5 resources

# Resource limit'leri azalt veya node ekle
```

**Çözüm 3: ConfigMap/Secret eksik**
```bash
# ConfigMap'leri kontrol et
kubectl get configmaps -n production

# Secret'ları kontrol et
kubectl get secrets -n production

# Eksikse oluştur
kubectl apply -f k8s/configmap.yaml -n production
kubectl apply -f k8s/secrets.yaml -n production
```

### Sorun: Service Erişilemiyor

**Tanı**:
```bash
# Service'i kontrol et
kubectl get svc -n production

# Endpoint'leri kontrol et
kubectl get endpoints -n production

# Service'e curl at
kubectl run curl-test --image=curlimages/curl -i --rm --restart=Never -- \
  curl http://backend-service:8000/health
```

**Çözüm 1: Selector'ları kontrol et**
```bash
# Service selector'ını al
kubectl get svc backend-service -n production -o yaml | grep -A 3 selector

# Pod label'larını al
kubectl get pods -n production --show-labels

# Eşleştiğinden emin ol
```

**Çözüm 2: Port mapping'i kontrol et**
```bash
# Service port'larını kontrol et
kubectl get svc backend-service -n production -o yaml | grep -A 5 ports

# Container port'unu kontrol et
kubectl get pod <pod-name> -n production -o yaml | grep containerPort
```

### Sorun: HPA Çalışmıyor

**Tanı**:
```bash
# HPA durumunu kontrol et
kubectl get hpa -n production

# HPA detaylarını görüntüle
kubectl describe hpa backend-hpa -n production

# Metrics server çalışıyor mu?
kubectl get deployment metrics-server -n kube-system
```

**Çözüm 1: Metrics server kur**
```bash
# Metrics server deploy et
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Çalıştığını doğrula
kubectl top nodes
kubectl top pods -n production
```

**Çözüm 2: Resource request'leri tanımla**
```yaml
# Deployment'ta resource request'ler olmalı
resources:
  requests:
    cpu: 500m
    memory: 1Gi
```

## Debug Araçları

### Docker Debug Komutları

```bash
# Container'a shell ile gir
docker-compose exec backend bash

# Container'ı inspect et
docker inspect <container-id>

# Container process'lerini görüntüle
docker top <container-id>

# Container resource kullanımı
docker stats <container-id>

# Container filesystem'ini incele
docker diff <container-id>

# Container'dan dosya kopyala
docker cp <container-id>:/app/logs/app.log ./
```

### Kubernetes Debug Komutları

```bash
# Pod'a shell ile gir
kubectl exec -it <pod-name> -n production -- bash

# Pod loglarını görüntüle
kubectl logs <pod-name> -n production

# Önceki container'ın logları
kubectl logs <pod-name> -n production --previous

# Pod'u debug modda çalıştır
kubectl debug <pod-name> -n production -it --image=busybox

# Port forward
kubectl port-forward <pod-name> 8000:8000 -n production

# Pod'dan dosya kopyala
kubectl cp <pod-name>:/app/logs/app.log ./app.log -n production
```

### Network Debug

```bash
# DNS çözümlemesini test et
docker-compose exec backend nslookup postgres

# Port connectivity test et
docker-compose exec backend nc -zv postgres 5432

# HTTP request test et
docker-compose exec backend curl -v http://backend:8000/health

# Network trace
docker-compose exec backend tcpdump -i any port 8000
```

### Performance Debug

```bash
# CPU profiling
docker-compose exec backend py-spy top --pid 1

# Memory profiling
docker-compose exec backend py-spy dump --pid 1

# Strace
docker-compose exec backend strace -p 1

# I/O monitoring
docker-compose exec backend iotop
```

## Yardım Alma

Sorun çözemediyseniz:

1. **Logları toplayın**:
```bash
# Docker logs
docker-compose logs > logs.txt

# Kubernetes logs
kubectl logs <pod-name> -n production > pod-logs.txt
kubectl describe pod <pod-name> -n production > pod-describe.txt
```

2. **Sistem bilgilerini toplayın**:
```bash
# Docker version
docker version > system-info.txt
docker-compose version >> system-info.txt

# Kubernetes version
kubectl version >> system-info.txt

# OS info
uname -a >> system-info.txt
```

3. **Issue açın**:
- Sorun açıklaması
- Beklenen davranış
- Gerçekleşen davranış
- Repro adımları
- Log dosyaları
- Sistem bilgileri

## Ek Kaynaklar

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [BuildKit Documentation](https://github.com/moby/buildkit)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
