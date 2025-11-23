# Trendyol Gift Recommendation - Backend

FastAPI tabanlı backend API servisi. TRM modelini kullanarak kişiselleştirilmiş hediye önerileri sunar.

## Özellikler

- FastAPI ile yüksek performanslı REST API
- PyTorch ile model inference
- Redis ile caching
- PostgreSQL ile veri saklama
- Celery ile asenkron task yönetimi
- Prometheus metrikleri
- Comprehensive logging

## Kurulum

### Gereksinimler

- Python 3.10+
- PostgreSQL 15+
- Redis 7+

### Yerel Geliştirme

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements-dev.txt

# Environment variables ayarla
cp .env.example .env
# .env dosyasını düzenle

# Veritabanı migration'larını çalıştır
alembic upgrade head

# Geliştirme sunucusunu başlat
uvicorn app.main:app --reload
```

## Test

```bash
# Tüm testleri çalıştır
pytest

# Coverage ile
pytest --cov=app --cov-report=html

# Sadece unit testler
pytest tests/unit

# Sadece property-based testler
pytest tests/property -v
```

## API Dokümantasyonu

Sunucu çalışırken:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Proje Yapısı

```
backend/
├── app/
│   ├── main.py              # FastAPI uygulaması
│   ├── config.py            # Konfigürasyon
│   ├── api/                 # API endpoints
│   ├── services/            # Business logic
│   ├── models/              # Database models
│   ├── schemas/             # Pydantic schemas
│   └── utils/               # Yardımcı fonksiyonlar
├── tests/
│   ├── unit/                # Unit testler
│   ├── property/            # Property-based testler
│   └── integration/         # Integration testler
└── alembic/                 # Database migrations
```
