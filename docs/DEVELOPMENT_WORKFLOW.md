# Geliştirme İş Akışı

Bu doküman, günlük geliştirme sürecinde kod değişiklikleri yapma, bağımlılık ekleme ve servisleri yönetme konularında rehberlik sağlar.

## İçindekiler

- [Kod Değişiklikleri Yapma](#kod-değişiklikleri-yapma)
- [Bağımlılık Ekleme](#bağımlılık-ekleme)
- [Servisleri Yeniden Başlatma](#servisleri-yeniden-başlatma)
- [Test Çalıştırma](#test-çalıştırma)
- [Debugging](#debugging)
- [En İyi Uygulamalar](#en-iyi-uygulamalar)

## Kod Değişiklikleri Yapma

### Hot Reload Nasıl Çalışır?

Development ortamında, kod değişiklikleriniz otomatik olarak container'a yansır ve uygulama yeniden yüklenir.

**Backend (Python/FastAPI)**
- Dosya değişikliği algılandığında uvicorn otomatik olarak yeniden yüklenir
- Değişiklikler 1-2 saniye içinde aktif olur
- Terminal'de reload mesajını göreceksiniz

**Frontend (React/Vite)**
- Hot Module Replacement (HMR) aktiftir
- Değişiklikler anında tarayıcıda görünür
- Sayfa durumu korunur (çoğu durumda)

### Backend Kod Değişiklikleri

1. **Python dosyasını düzenleyin**:
```bash
# Örnek: backend/app/api/recommendations.py
nano backend/app/api/recommendations.py
```

2. **Değişikliği kaydedin** - Otomatik reload başlar

3. **Terminal'de reload mesajını kontrol edin**:
```bash
docker-compose logs -f backend
```

Beklenen çıktı:
```
backend_1  | INFO:     Uvicorn running on http://0.0.0.0:8000
backend_1  | INFO:     Application startup complete.
backend_1  | WARNING:  StatReload detected changes in 'app/api/recommendations.py'. Reloading...
```

4. **API'yi test edin**:
```bash
curl http://localhost:8000/api/recommendations
```

### Frontend Kod Değişiklikleri

1. **React component'ini düzenleyin**:
```bash
# Örnek: frontend/src/components/RecommendationCard.tsx
nano frontend/src/components/RecommendationCard.tsx
```

2. **Değişikliği kaydedin** - HMR otomatik çalışır

3. **Tarayıcıyı kontrol edin** - Değişiklik anında görünür

4. **Console'da HMR mesajını görün**:
```
[vite] hot updated: /src/components/RecommendationCard.tsx
```

### Configuration Dosyası Değişiklikleri

Configuration dosyaları değiştiğinde sadece ilgili servisler yeniden başlar:

```bash
# Backend config değişti - sadece backend restart olur
nano backend/.env
docker-compose restart backend

# Frontend config değişti - sadece frontend restart olur
nano frontend/.env
docker-compose restart frontend
```

## Bağımlılık Ekleme

### Backend Bağımlılık Ekleme (Python)

1. **requirements.txt dosyasını güncelleyin**:
```bash
nano backend/requirements.txt
```

Yeni paketi ekleyin:
```
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
redis==5.0.1
httpx==0.25.2  # YENİ PAKET
```

2. **Sadece dependency layer'ını yeniden build edin**:
```bash
docker-compose build backend
```

**Önemli**: Cache optimizasyonu sayesinde sadece dependency layer yeniden build edilir (~30 saniye).

3. **Servisi yeniden başlatın**:
```bash
docker-compose up -d backend
```

4. **Yeni paketi kullanın**:
```python
# backend/app/services/new_service.py
import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()
```

### Frontend Bağımlılık Ekleme (npm)

1. **package.json'ı güncelleyin veya npm install kullanın**:

**Yöntem 1: package.json'ı manuel düzenle**
```bash
nano frontend/package.json
```

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.2"  // YENİ PAKET
  }
}
```

**Yöntem 2: Container içinde npm install**
```bash
docker-compose exec frontend npm install axios
```

2. **Dependency layer'ını yeniden build edin**:
```bash
docker-compose build frontend
```

3. **Servisi yeniden başlatın**:
```bash
docker-compose up -d frontend
```

4. **Yeni paketi kullanın**:
```typescript
// frontend/src/services/api.ts
import axios from 'axios';

export const fetchRecommendations = async () => {
  const response = await axios.get('/api/recommendations');
  return response.data;
};
```

### Development Dependencies Ekleme

**Backend (requirements-dev.txt)**:
```bash
nano backend/requirements-dev.txt
```

```
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
mypy==1.7.1
```

**Frontend (devDependencies)**:
```bash
docker-compose exec frontend npm install --save-dev @types/node
```

## Servisleri Yeniden Başlatma

### Tek Servis Yeniden Başlatma

```bash
# Backend'i yeniden başlat
docker-compose restart backend

# Frontend'i yeniden başlat
docker-compose restart frontend

# PostgreSQL'i yeniden başlat
docker-compose restart postgres

# Redis'i yeniden başlat
docker-compose restart redis
```

**Not**: Diğer servisler çalışmaya devam eder.

### Tüm Servisleri Yeniden Başlatma

```bash
docker-compose restart
```

### Servis Durdurma ve Başlatma

```bash
# Servisi durdur
docker-compose stop backend

# Servisi başlat
docker-compose start backend

# Veya her ikisini birden
docker-compose restart backend
```

### Yeniden Build ile Başlatma

Dockerfile veya build context değiştiğinde:

```bash
# Belirli bir servisi rebuild et ve başlat
docker-compose up -d --build backend

# Tüm servisleri rebuild et ve başlat
docker-compose up -d --build
```

### Cache Kullanmadan Build

Sorun yaşıyorsanız cache'i temizleyin:

```bash
# Belirli bir servis için
docker-compose build --no-cache backend

# Tüm servisler için
docker-compose build --no-cache
```

## Test Çalıştırma

### Backend Testleri

**Tüm testleri çalıştır**:
```bash
docker-compose exec backend pytest
```

**Belirli bir test dosyasını çalıştır**:
```bash
docker-compose exec backend pytest tests/unit/test_recommendations.py
```

**Coverage ile çalıştır**:
```bash
docker-compose exec backend pytest --cov=app --cov-report=html
```

**Property-based testleri çalıştır**:
```bash
docker-compose exec backend pytest tests/property/ -v
```

### Frontend Testleri

**Tüm testleri çalıştır**:
```bash
docker-compose exec frontend npm test
```

**Watch mode ile çalıştır**:
```bash
docker-compose exec frontend npm test -- --watch
```

**Coverage ile çalıştır**:
```bash
docker-compose exec frontend npm test -- --coverage
```

## Debugging

### Backend Debugging

**1. Logları görüntüle**:
```bash
# Tüm loglar
docker-compose logs -f backend

# Son 100 satır
docker-compose logs --tail=100 backend

# Belirli bir zaman aralığı
docker-compose logs --since 10m backend
```

**2. Interactive shell**:
```bash
docker-compose exec backend bash
```

**3. Python debugger (pdb)**:

Kodunuza breakpoint ekleyin:
```python
def my_function():
    import pdb; pdb.set_trace()  # Breakpoint
    # kod devam eder
```

Container'ı interactive modda çalıştırın:
```bash
docker-compose run --service-ports backend
```

**4. Environment değişkenlerini kontrol et**:
```bash
docker-compose exec backend env
```

### Frontend Debugging

**1. Browser DevTools**:
- Chrome/Firefox DevTools'u açın (F12)
- Console, Network, Sources tab'lerini kullanın

**2. Logları görüntüle**:
```bash
docker-compose logs -f frontend
```

**3. Build hatalarını kontrol et**:
```bash
docker-compose exec frontend npm run build
```

**4. Node container'ına gir**:
```bash
docker-compose exec frontend sh
```

### Database Debugging

**PostgreSQL'e bağlan**:
```bash
docker-compose exec postgres psql -U postgres -d mydb
```

**Tabloları listele**:
```sql
\dt
```

**Query çalıştır**:
```sql
SELECT * FROM users LIMIT 10;
```

**Redis'e bağlan**:
```bash
docker-compose exec redis redis-cli
```

**Key'leri listele**:
```
KEYS *
```

**Değer oku**:
```
GET mykey
```

## En İyi Uygulamalar

### 1. Küçük, Sık Commit'ler

```bash
# Her özellik için ayrı commit
git add backend/app/api/new_endpoint.py
git commit -m "feat: add new recommendation endpoint"

# Test'leri ayrı commit
git add backend/tests/test_new_endpoint.py
git commit -m "test: add tests for new endpoint"
```

### 2. Branch Stratejisi

```bash
# Yeni özellik için branch oluştur
git checkout -b feature/recommendation-algorithm

# Değişiklikleri yap ve commit et
git add .
git commit -m "feat: implement new algorithm"

# Main branch'e merge et
git checkout main
git merge feature/recommendation-algorithm
```

### 3. Düzenli Cache Temizliği

```bash
# Haftada bir kez
docker system prune -a --volumes

# Veya sadece kullanılmayan image'ları
docker image prune -a
```

### 4. Log Seviyelerini Kullan

**Development**:
```bash
# .env dosyasında
LOG_LEVEL=DEBUG
```

**Production**:
```bash
LOG_LEVEL=INFO
```

### 5. Health Check'leri Kontrol Et

```bash
# Düzenli olarak kontrol et
docker-compose ps

# Health endpoint'i test et
curl http://localhost:8000/health
```

### 6. Bağımlılıkları Güncel Tut

```bash
# Backend
docker-compose exec backend pip list --outdated

# Frontend
docker-compose exec frontend npm outdated
```

### 7. Test Coverage'ı İzle

```bash
# Backend coverage
docker-compose exec backend pytest --cov=app --cov-report=term-missing

# Frontend coverage
docker-compose exec frontend npm test -- --coverage
```

## Yaygın Geliştirme Senaryoları

### Senaryo 1: Yeni API Endpoint Ekleme

1. Backend'de endpoint'i oluştur
2. Hot reload ile test et
3. Frontend'de API çağrısı ekle
4. HMR ile tarayıcıda test et
5. Test yaz ve çalıştır
6. Commit et

### Senaryo 2: Database Schema Değişikliği

1. Migration dosyası oluştur
2. Container'da migration çalıştır:
```bash
docker-compose exec backend alembic upgrade head
```
3. Model'leri güncelle
4. Test et
5. Commit et

### Senaryo 3: Yeni Servis Ekleme

1. docker-compose.yml'e servis ekle
2. Dockerfile oluştur
3. Build et:
```bash
docker-compose build new-service
```
4. Başlat:
```bash
docker-compose up -d new-service
```
5. Test et ve commit et

## Performans İpuçları

### Build Süresini Azaltma

1. **.dockerignore'u optimize et** - Gereksiz dosyaları hariç tut
2. **Layer sırasını optimize et** - Sık değişen dosyaları sona koy
3. **BuildKit cache'i kullan** - `DOCKER_BUILDKIT=1`
4. **Multi-stage build kullan** - Sadece gerekli dosyaları kopyala

### Hot Reload Performansı

1. **Büyük dosyaları hariç tut** - node_modules, __pycache__
2. **Named volume kullan** - Bağımlılıklar için
3. **Bind mount optimize et** - Sadece kaynak kodu mount et

### Container Kaynak Kullanımı

```bash
# Kaynak kullanımını izle
docker stats

# Belirli bir container için
docker stats backend_1
```

## Sorun Giderme

Sorunlarla karşılaşırsanız [Sorun Giderme Kılavuzu](./TROUBLESHOOTING.md)'na bakın.

## Sonraki Adımlar

- [Production Deployment](./PRODUCTION_DEPLOYMENT.md) - Production'a nasıl deploy edilir
- [Sorun Giderme](./TROUBLESHOOTING.md) - Yaygın sorunlar ve çözümleri
- [API Dokümantasyonu](./API_DOCUMENTATION.md) - API endpoint'leri ve kullanımı
