# Trendyol Gift Recommendation System

Eğitilmiş TinyRecursiveModels (TRM) modelini kullanarak Trendyol üzerinden gerçek zamanlı, kişiselleştirilmiş hediye önerileri sunan full-stack web uygulaması.

## 🎯 Özellikler

- **Kişiselleştirilmiş Öneriler**: Kullanıcı profiline göre özel hediye önerileri
- **Çoklu Analiz Araçları**: 6 farklı analiz aracı ile kapsamlı ürün değerlendirmesi
- **Gerçek Zamanlı Veri**: Trendyol web scraping entegrasyonu ile güncel ürün bilgileri
- **Modern UI/UX**: React ve Tailwind CSS ile responsive tasarım
- **Yüksek Performans**: Redis caching ve optimize edilmiş model inference
- **Dark Mode**: Göz dostu karanlık tema desteği

## ⚠️ Önemli: Trendyol API → Scraping Geçişi

Trendyol'un gerçek bir API'si olmadığı için backend servisi **web scraping** tabanlı bir implementasyona dönüştürüldü. 

📖 **Detaylı bilgi için**: [`backend/QUICKSTART.md`](backend/QUICKSTART.md)

**Hızlı kurulum**:
```bash
cd backend
pip install -r requirements.txt
playwright install chromium
python tests/test_trendyol_scraping.py  # Test
```


## 🏗️ Mimari

```
┌─────────────┐
│   Frontend  │  React + TypeScript + Tailwind
│   (Port 3000)│
└──────┬──────┘
       │ REST API
       ▼
┌─────────────┐
│   Backend   │  FastAPI + PyTorch
│   (Port 8000)│
└──────┬──────┘
       │
       ├──────► PostgreSQL (Database)
       ├──────► Redis (Cache)
       └──────► Trendyol (Web Scraping via Playwright)
```

**Not**: Backend artık Trendyol'dan veri çekmek için web scraping kullanıyor (Trendyol'un API'si olmadığı için).


## 🚀 Hızlı Başlangıç

### Gereksinimler

- Docker & Docker Compose
- Node.js 20+ (yerel geliştirme için)
- Python 3.10+ (yerel geliştirme için)

### Docker ile Başlatma (Önerilen)

```bash
# Environment dosyalarını oluştur
make setup-env

# .env dosyalarını düzenle (özellikle Trendyol API anahtarları)
nano .env

# Tüm servisleri başlat
make dev

# veya
docker-compose up -d
```

Servisler:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Yerel Geliştirme

```bash
# Projeyi başlat
make init

# Backend'i başlat
cd backend
source venv/bin/activate
uvicorn app.main:app --reload

# Frontend'i başlat (yeni terminal)
cd frontend
npm run dev
```

## 📁 Proje Yapısı

```
.
├── frontend/              # React frontend uygulaması
│   ├── src/
│   ├── public/
│   └── package.json
├── backend/               # FastAPI backend uygulaması
│   ├── app/
│   ├── tests/
│   └── requirements.txt
├── models/                # TRM model dosyaları
├── checkpoints/           # Eğitilmiş model checkpoint'leri
├── docker-compose.yml     # Development ortamı
├── docker-compose.prod.yml # Production ortamı
└── Makefile              # Yardımcı komutlar
```

## 🧪 Test

```bash
# Tüm testleri çalıştır
make test

# Sadece backend testleri
make test-backend

# Sadece frontend testleri
make test-frontend
```

## 🔧 Geliştirme

### Kod Kalitesi

```bash
# Linting
make lint

# Formatting
make format

# Pre-commit hooks'ları yükle
pre-commit install
```

### Docker Komutları

```bash
# Servisleri başlat
make docker-up

# Servisleri durdur
make docker-down

# Logları görüntüle
make docker-logs

# Temizlik
make docker-clean
```

## 📊 Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## 🔐 Güvenlik

- HTTPS zorunluluğu
- Rate limiting
- Input sanitization
- XSS ve SQL injection koruması
- Veri şifreleme
- JWT authentication

## 📝 API Dokümantasyonu

Backend çalışırken:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- TinyRecursiveModels (TRM) modeli
- Trendyol API
- Açık kaynak topluluğu

## 📧 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.
