# Performance Testing Quick Reference

## Hızlı Başlangıç

### Tüm Testleri Çalıştır

```bash
# Otomatik validation script
./scripts/validate-performance.sh

# Veya pytest ile
cd backend
pytest tests/performance/test_container_performance.py -v
```

## Test Kategorileri

### 1. Build Time Tests (14.1)

**Hedefler:**
- Cold build: < 10 dakika
- Warm build: < 2 dakika  
- Incremental build: < 30 saniye

**Çalıştırma:**
```bash
pytest tests/performance/test_container_performance.py::TestBuildPerformance -v
```

### 2. Image Size Tests (14.2)

**Hedefler:**
- Backend: < 200MB
- Frontend: < 50MB

**Çalıştırma:**
```bash
pytest tests/performance/test_container_performance.py::TestImageSizes -v
```

### 3. Hot Reload Tests (14.3)

**Hedef:** < 2 saniye

**Manuel Test:**
```bash
docker-compose up -d
# Kod değişikliği yap ve log'ları izle
docker-compose logs -f backend
```

### 4. Cache Efficiency Tests (14.4)

**Hedef:** > %95 cache hit rate

**Çalıştırma:**
```bash
pytest tests/performance/test_container_performance.py::TestCacheEfficiency -v
```

## Yavaş Testleri Atlama

```bash
# Yavaş testleri atla
pytest tests/performance/test_container_performance.py -v -m "not slow"

# Veya environment variable ile
SKIP_SLOW_TESTS=1 pytest tests/performance/test_container_performance.py -v
```

## Detaylı Doküman

Daha fazla bilgi için: [docs/PERFORMANCE_VALIDATION.md](docs/PERFORMANCE_VALIDATION.md)
