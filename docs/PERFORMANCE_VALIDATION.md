# Container Infrastructure Performance Validation

Bu doküman, konteyner altyapısının performans validasyonunu açıklar.

## Genel Bakış

Performance validation, aşağıdaki metrikleri test eder:

1. **Build Times** (Task 14.1)
   - Cold build time: < 10 dakika
   - Warm build time: < 2 dakika
   - Incremental build time: < 30 saniye

2. **Image Sizes** (Task 14.2)
   - Backend image: < 200MB
   - Frontend image: < 50MB

3. **Hot Reload Performance** (Task 14.3)
   - Code change to reload: < 2 saniye

4. **Cache Efficiency** (Task 14.4)
   - Cache hit rate: > %95 (değişmeyen bağımlılıklar için)

## Gereksinimler

- Docker ve Docker Compose kurulu olmalı
- BuildKit etkin olmalı
- Python 3.8+ (script çalıştırmak için)
- pytest (test suite için)

## Kullanım

### Otomatik Validation Script

Tüm performans testlerini çalıştırmak için:

```bash
# Shell script ile
./scripts/validate-performance.sh

# Veya doğrudan Python ile
python3 scripts/validate-performance.py
```

Script şunları yapar:
- Docker'ın çalıştığını kontrol eder
- BuildKit'in etkin olduğunu doğrular
- Tüm performans testlerini çalıştırır
- Detaylı rapor oluşturur

### Pytest Test Suite

Backend test suite'i kullanarak:

```bash
# Tüm performans testlerini çalıştır
cd backend
pytest tests/performance/test_container_performance.py -v

# Yavaş testleri atla
pytest tests/performance/test_container_performance.py -v -m "not slow"

# Sadece belirli bir test sınıfı
pytest tests/performance/test_container_performance.py::TestBuildPerformance -v

# Sadece belirli bir test
pytest tests/performance/test_container_performance.py::TestImageSizes::test_backend_image_size -v
```

### Manuel Validation

#### Build Times (Task 14.1)

**Cold Build Test:**
```bash
# Backend
time docker build --no-cache -f ./backend/Dockerfile --target production -t backend:cold-test ./backend

# Frontend
time docker build --no-cache -f ./frontend/Dockerfile --target production -t frontend:cold-test ./frontend
```

**Warm Build Test:**
```bash
# İlk build (cache oluştur)
docker build -f ./backend/Dockerfile --target production -t backend:warm-test-1 ./backend

# İkinci build (cache kullan)
time docker build -f ./backend/Dockerfile --target production -t backend:warm-test-2 ./backend
```

**Incremental Build Test:**
```bash
# İlk build
docker build -f ./backend/Dockerfile --target production -t backend:inc-test-1 ./backend

# Kod değişikliği yap
echo "# Test comment" >> backend/app/__init__.py

# Rebuild
time docker build -f ./backend/Dockerfile --target production -t backend:inc-test-2 ./backend

# Değişikliği geri al
git checkout backend/app/__init__.py
```

#### Image Sizes (Task 14.2)

```bash
# Backend image boyutu
docker build -f ./backend/Dockerfile --target production -t backend:size-test ./backend
docker image inspect backend:size-test --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'

# Frontend image boyutu
docker build -f ./frontend/Dockerfile --target production -t frontend:size-test ./frontend
docker image inspect frontend:size-test --format='{{.Size}}' | awk '{print $1/1024/1024 " MB"}'
```

#### Hot Reload Performance (Task 14.3)

```bash
# Development ortamını başlat
docker-compose up -d

# Backend'de değişiklik yap ve log'ları izle
docker-compose logs -f backend

# Kod değişikliği yap (başka bir terminalde)
echo "# Test" >> backend/app/main.py

# Reload süresini log'lardan ölç
# Beklenen: < 2 saniye
```

#### Cache Efficiency (Task 14.4)

```bash
# İlk build
docker build -f ./backend/Dockerfile --target production -t backend:cache-1 ./backend

# İkinci build (cache analizi)
docker build -f ./backend/Dockerfile --target production -t backend:cache-2 ./backend 2>&1 | grep -i "cached"

# Cache hit rate'i hesapla
# Beklenen: > %95 cached steps
```

## Çıktı ve Raporlama

### Script Çıktısı

Validation script şu formatta çıktı verir:

```
[HH:MM:SS] INFO: Starting Performance Validation Suite
[HH:MM:SS] INFO: ============================================================
[HH:MM:SS] INFO: TASK 14.1: Validating Build Times
[HH:MM:SS] INFO: ============================================================

[HH:MM:SS] INFO: 1. Testing cold build time (backend)...
[HH:MM:SS] SUCCESS: ✓ Cold build: 245.32s < 600s

[HH:MM:SS] INFO: 2. Testing warm build time (backend)...
[HH:MM:SS] SUCCESS: ✓ Warm build: 45.12s < 120s

...

[HH:MM:SS] INFO: ============================================================
[HH:MM:SS] INFO: PERFORMANCE VALIDATION REPORT
[HH:MM:SS] INFO: ============================================================

[HH:MM:SS] INFO: Total Tests: 8
[HH:MM:SS] SUCCESS: Passed: 8
[HH:MM:SS] ERROR: Failed: 0
[HH:MM:SS] INFO: Pass Rate: 100.0%

[HH:MM:SS] INFO: Detailed report saved to: performance-validation-report.json
```

### JSON Rapor

Detaylı sonuçlar `performance-validation-report.json` dosyasına kaydedilir:

```json
{
  "timestamp": "2024-11-26T10:30:00",
  "summary": {
    "total": 8,
    "passed": 8,
    "failed": 0,
    "pass_rate": 100.0
  },
  "results": {
    "backend_cold_build": {
      "time": 245.32,
      "target": 600,
      "passed": true
    },
    "backend_image_size": {
      "size": 157286400,
      "size_mb": 150.0,
      "target": 209715200,
      "target_mb": 200,
      "passed": true
    }
  },
  "targets": {
    "cold_build_time": 600,
    "warm_build_time": 120,
    "incremental_build_time": 30,
    "backend_image_size": 209715200,
    "frontend_image_size": 52428800,
    "hot_reload_latency": 2,
    "cache_hit_rate": 0.95
  }
}
```

## Troubleshooting

### Docker Build Yavaş

**Problem:** Build süreleri hedefleri aşıyor

**Çözümler:**
1. BuildKit'in etkin olduğunu kontrol edin:
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```

2. Docker cache'i temizleyin ve yeniden deneyin:
   ```bash
   docker builder prune -a
   ```

3. .dockerignore dosyasının doğru yapılandırıldığını kontrol edin

4. Layer sıralamasını optimize edin (sık değişenler en sonda)

### Image Boyutu Büyük

**Problem:** Image boyutları hedefleri aşıyor

**Çözümler:**
1. Multi-stage build kullanıldığını doğrulayın
2. Production stage'de sadece runtime dependencies olduğunu kontrol edin
3. Gereksiz dosyaların temizlendiğini doğrulayın:
   ```dockerfile
   RUN find /app -type d -name "__pycache__" -exec rm -rf {} + && \
       find /app -type f -name "*.pyc" -delete
   ```

4. Alpine base image kullanın

### Cache Çalışmıyor

**Problem:** Cache hit rate düşük

**Çözümler:**
1. Dependency dosyalarının önce kopyalandığını kontrol edin:
   ```dockerfile
   COPY requirements.txt ./
   RUN pip install -r requirements.txt
   COPY . .
   ```

2. BuildKit cache mount kullanın:
   ```dockerfile
   RUN --mount=type=cache,target=/root/.cache/pip \
       pip install -r requirements.txt
   ```

3. .dockerignore'da gereksiz dosyaların exclude edildiğini doğrulayın

### Hot Reload Çalışmıyor

**Problem:** Kod değişiklikleri yansımıyor

**Çözümler:**
1. Volume mount'ların doğru yapılandırıldığını kontrol edin:
   ```yaml
   volumes:
     - ./backend:/app
     - /app/__pycache__  # Exclude
   ```

2. Development target kullanıldığını doğrulayın:
   ```yaml
   build:
     target: development
   ```

3. Hot reload komutunun çalıştığını kontrol edin:
   ```dockerfile
   CMD ["uvicorn", "app.main:app", "--reload"]
   ```

## CI/CD Entegrasyonu

### GitHub Actions

```yaml
name: Performance Validation

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Run Performance Validation
        run: |
          chmod +x scripts/validate-performance.sh
          ./scripts/validate-performance.sh
      
      - name: Upload Performance Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: performance-validation-report.json
```

## Performans Hedefleri

| Metrik | Hedef | Gereksinim |
|--------|-------|------------|
| Cold Build Time | < 10 dakika | 1.1, 2.1 |
| Warm Build Time | < 2 dakika | 2.1 |
| Incremental Build Time | < 30 saniye | 2.4 |
| Backend Image Size | < 200MB | 3.1 |
| Frontend Image Size | < 50MB | 3.2 |
| Hot Reload Latency | < 2 saniye | 2.1, 2.2 |
| Cache Hit Rate | > %95 | 4.2, 9.1 |

## İlgili Dokümanlar

- [Design Document](../.kiro/specs/optimized-container-infrastructure/design.md)
- [Requirements Document](../.kiro/specs/optimized-container-infrastructure/requirements.md)
- [Tasks Document](../.kiro/specs/optimized-container-infrastructure/tasks.md)
- [Quick Start Guide](./QUICK_START.md)
- [Development Workflow](./DEVELOPMENT_WORKFLOW.md)
