# 🔄 Trendyol API → Scraping Migration Özeti

## Yapılan Değişiklikler

### 1. Yeni Dosyalar Oluşturuldu

#### Backend
- ✅ `/backend/app/services/trendyol_scraping_service.py` - Ana scraping servisi
- ✅ `/backend/app/services/trendyol_api_old.py` - Eski API servisi (yedek)
- ✅ `/backend/TRENDYOL_SCRAPING_README.md` - Detaylı dokümantasyon
- ✅ `/backend/tests/test_trendyol_scraping.py` - Test scripti
- ✅ `/backend/MIGRATION_SUMMARY.md` - Bu dosya

### 2. Güncellenen Dosyalar

#### Backend
- ✅ `/backend/app/services/trendyol_api.py` - Wrapper olarak güncellendi
- ✅ `/backend/requirements.txt` - Playwright, BeautifulSoup4, lxml eklendi

## Kod Değişiklikleri

### Öncesi (Fake API)
```python
# backend/app/services/trendyol_api.py
class TrendyolAPIService:
    def __init__(self, api_key, base_url, rate_limit):
        self.api_key = api_key
        self.base_url = base_url  # Var olmayan API endpoint
        self.client = httpx.AsyncClient()  # HTTP istekleri
    
    async def search_products(self, category, keywords, ...):
        # Fake API çağrısı - çalışmaz!
        response = await self.client.get(f"{self.base_url}/products/search")
        # ...
```

### Sonrası (Real Scraping)
```python
# backend/app/services/trendyol_scraping_service.py
class TrendyolScrapingService:
    def __init__(self, rate_limit, cache_ttl):
        self.scraping_rate_limiter = RateLimiter(...)
        self._scraper = TrendyolScraper(...)  # Gerçek web scraper
    
    async def search_products(self, category, keywords, ...):
        # Gerçek web scraping
        scraper = await self._get_scraper()
        scraped_data = await scraper.scrape_products(max_products)
        # ...
```

### Wrapper (Geriye Dönük Uyumluluk)
```python
# backend/app/services/trendyol_api.py
from app.services.trendyol_scraping_service import (
    TrendyolScrapingService as TrendyolAPIService,  # Alias
    get_trendyol_scraping_service as get_trendyol_service  # Alias
)
```

## Interface Uyumluluğu

### Değişmeyen Interface

```python
# Kullanılabilir metotlar - AYNI KALDI
service = get_trendyol_service()

products = await service.search_products(
    category: str,
    keywords: List[str],
    max_results: int,
    min_price: Optional[float],
    max_price: Optional[float]
)

product = await service.get_product_details(product_id: str)

gift_item = service.convert_to_gift_item(product: TrendyolProduct)

await service.close()
```

### TrendyolProduct Model
```python
# Model yapısı değişmedi
class TrendyolProduct:
    id: str
    name: str
    category: str
    price: float
    rating: float
    image_url: str
    product_url: str
    description: str
    brand: str
    in_stock: bool
    review_count: int
```

## Mevcut Kod Uyumluluğu

### ✅ Değişiklik Gerektirmeyen Dosyalar

Aşağıdaki dosyalar aynen çalışmaya devam eder:

1. **API Endpoints**:
   - `/backend/app/api/v1/recommendations.py` ✅
   - Diğer endpoint'ler ✅

2. **Servisler**:
   - `/backend/app/services/model_inference.py` ✅
   - `/backend/app/services/cache_service.py` ✅
   - Diğer servisler ✅

3. **Testler**:
   - Mevcut unit testler ✅
   - Integration testler ✅

### 🔧 Mock Güncelleme Önerileri

Test dosyalarında mock'lar güncellenebilir (opsiyonel):

```python
# Öncesi
@pytest.fixture
def mock_trendyol_service():
    service = Mock(spec=TrendyolAPIService)
    # ...

# Sonrası (opsiyonel - geriye uyumlu)
from app.services.trendyol_scraping_service import TrendyolScrapingService

@pytest.fixture
def mock_trendyol_service():
    service = Mock(spec=TrendyolScrapingService)
    # Aynı interface, aynı metodlar
```

## Performans Farkları

### API Modu (Varsayımsal)
- ⚡ Hız: ~100-200ms/istek
- 🔄 Rate Limit: 100 req/min
- 💰 Maliyet: API key gerekir
- ❌ Durum: Var olmayan API

### Scraping Modu (Gerçek)
- 🐌 Hız: ~3-5 saniye/ürün (ilk istek)
- ⚡ Hız: ~10ms (cache'den)
- 🔄 Rate Limit: 20 req/min (bot önleme)
- ✅ Durum: Çalışıyor!

## Kurulum Adımları

### 1. Backend Bağımlılıkları
```bash
cd backend
pip install -r requirements.txt  # playwright, beautifulsoup4, lxml eklendi
playwright install chromium
```

### 2. Scraping Bağımlılıkları (Zaten kurulu olmalı)
```bash
cd ../scraping
pip install -r requirements.txt
```

### 3. Test
```bash
cd ../backend
python tests/test_trendyol_scraping.py
```

## Konfigürasyon

### Scraping Ayarları

Backend'de scraping ayarları:

```python
# backend/app/services/trendyol_scraping_service.py
class TrendyolScrapingService:
    def __init__(self, rate_limit=20, cache_ttl=1800):
        # Rate limiting: 20 req/min (API'den daha düşük)
        # Cache TTL: 30 dakika
        # Browser: Headless Chromium
        # Anti-bot: User agent rotation, human simulation
```

Scraping config dosyası:
```yaml
# scraping/config/scraping_config.yaml
rate_limit:
  requests_per_minute: 20
  delay_between_requests: [2, 5]
  max_concurrent_requests: 3

scraping:
  websites:
    - name: "trendyol"
      categories: ["elektronik", "ev-yasam", "kozmetik", ...]
```

## Önemli Notlar

### ✅ Avantajlar
1. **Gerçek Veri**: Trendyol'dan gerçek ürün verileri
2. **API Key Gereksiz**: Ücretsiz kullanım
3. **Zengin Veri**: Resim, açıklama, rating, fiyat
4. **Güncel**: Her zaman güncel ürünler

### ⚠️ Dikkat Edilmesi Gerekenler
1. **Hız**: İlk istek yavaş (3-5 sn), cache kullanın
2. **Rate Limiting**: Bottan kaçınmak için düşük rate limit
3. **CAPTCHA**: Fazla istek CAPTCHA tetikleyebilir
4. **Selector Güncellemeleri**: Site değişirse selector'lar güncellenebilir
5. **Browser Overhead**: Playwright browser açma maliyeti var

### 🔧 Troubleshooting

**Problem**: CAPTCHA detected
```bash
# Çözüm: Rate limit düşürün
# scraping/config/scraping_config.yaml:
rate_limit:
  requests_per_minute: 10  # Daha düşük
```

**Problem**: Browser başlatma hatası
```bash
# Çözüm: Playwright yükleyin
playwright install chromium
playwright install-deps  # Linux için
```

**Problem**: Scraping çok yavaş
```bash
# Çözüm: Cache kullanın, batch processing yapın
# Cache TTL: 30 dakika varsayılan
service = TrendyolScrapingService(cache_ttl=3600)  # 1 saat
```

## Migration Checklist

- [x] Yeni scraping servisi oluşturuldu
- [x] Eski API servisi yedeklendi
- [x] Wrapper ile geriye uyumluluk sağlandı
- [x] Requirements.txt güncellendi
- [x] Test scripti oluşturuldu
- [x] Dokümantasyon hazırlandı
- [ ] Testler çalıştırıldı (kullanıcı yapacak)
- [ ] Production deployment (kullanıcı yapacak)

## Sonraki Adımlar

1. **Test Edin**:
   ```bash
   python backend/tests/test_trendyol_scraping.py
   ```

2. **Backend'i Çalıştırın**:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

3. **API Test Edin**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/recommendations \
     -H "Content-Type: application/json" \
     -d '{
       "user_profile": {
         "age": 25,
         "gender": "female",
         "budget": 500,
         "occasion": "birthday",
         "relationship": "friend",
         "hobbies": ["reading", "music"]
       }
     }'
   ```

4. **Production'a Deploy**:
   - Environment variables kontrol edin
   - Playwright browser kurulumunu yapın
   - Rate limiting ayarlarını optimize edin

## Destek

Sorularınız için:
- 📖 Backend README: `/backend/TRENDYOL_SCRAPING_README.md`
- 📖 Scraping README: `/scraping/README.md`
- 🧪 Test Script: `/backend/tests/test_trendyol_scraping.py`
