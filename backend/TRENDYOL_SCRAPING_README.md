# Trendyol Scraping Entegrasyonu

## 🔄 Değişiklikler

Trendyol'un gerçek bir API'si olmadığı için, backend'deki `TrendyolAPIService` modülü **web scraping** tabanlı bir implementasyona dönüştürüldü.

### Yapılan Değişiklikler

1. **Yeni Scraping Servisi**: `trendyol_scraping_service.py` oluşturuldu
   - `scraping/scrapers/trendyol_scraper.py` modülünü kullanır
   - Mevcut API servisinin aynı interface'ini korur
   - Cache mekanizması korundu
   - Rate limiting eklendi (scraping için daha düşük)

2. **Geriye Dönük Uyumluluk**: `trendyol_api.py` wrapper olarak güncellendi
   - Eski import'lar çalışmaya devam eder
   - Mevcut kod değişiklik gerektirmez

3. **Bağımlılıklar**: `backend/requirements.txt` güncellendi
   - `playwright==1.40.0`
   - `beautifulsoup4==4.12.2`
   - `lxml==4.9.3`

## 🚀 Kurulum

### 1. Backend Bağımlılıklarını Yükleyin

```bash
cd backend
pip install -r requirements.txt
```

### 2. Playwright Browser'ı Kurun

```bash
playwright install chromium
```

### 3. Scraping Bağımlılıklarını Kontrol Edin

```bash
cd ../scraping
pip install -r requirements.txt
```

## 📖 Kullanım

### Kod Değişikliği Gerekmiyor

Mevcut backend kodu aynen çalışmaya devam eder! Servis otomatik olarak scraping kullanacak:

```python
from app.services.trendyol_api import get_trendyol_service

# Aynı API
service = get_trendyol_service()

# Ama şimdi scraping kullanıyor
products = await service.search_products(
    category="elektronik",
    keywords=["kulaklık"],
    max_results=20
)
```

### Servis Detayları

#### TrendyolScrapingService

Özellikler:
- ✅ **Cache**: 30 dakika TTL (varsayılan)
- ✅ **Rate Limiting**: Dakikada 20 istek (bottan kaçınmak için)
- ✅ **Browser Management**: Otomatik Playwright yönetimi
- ✅ **Anti-Bot**: User agent rotation, human simulation
- ✅ **Fallback**: Cache'den eski veri kullanma

Ana Metodlar:
```python
# Ürün arama
await service.search_products(
    category: str,
    keywords: List[str],
    max_results: int = 50,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
)

# Ürün detayı
await service.get_product_details(product_id: str)

# GiftItem'a dönüştürme
service.convert_to_gift_item(product: TrendyolProduct)

# Temizlik
await service.close()
```

## ⚙️ Konfigürasyon

### Rate Limiting

Scraping servisi için:
```python
service = TrendyolScrapingService(
    rate_limit=20,  # Dakikada max istek sayısı
    cache_ttl=1800  # Cache süresi (saniye)
)
```

### Kategori Mapping

Desteklenen kategoriler:
- `elektronik`
- `ev_yasam` / `ev`
- `kozmetik`
- `giyim` / `kadin` / `erkek`
- `cocuk`
- `ayakkabi`
- `supermarket`
- `mobilya`
- `spor`
- `kitap`

## 🔧 Sorun Giderme

### CAPTCHA Detected

Eğer scraping sırasında CAPTCHA ile karşılaşılırsa:
1. Rate limit'i düşürün
2. Delay süresini artırın
3. Biraz bekleyip tekrar deneyin

```python
# scraping/config/scraping_config.yaml içinde:
rate_limit:
  requests_per_minute: 10  # Daha düşük
  delay_between_requests: [3, 7]  # Daha uzun
```

### Browser Başlatma Hatası

Playwright browser kurulumu gerekli:
```bash
playwright install chromium
playwright install-deps  # Linux'ta sistem bağımlılıkları için
```

### Import Error

`scraping` klasörü bulunamıyor hatası:
- Scraping klasörü backend'in parent dizininde olmalı
- Path otomatik olarak ekleniyor: `SCRAPING_DIR = Path(__file__).parent.parent.parent.parent / "scraping"`

### Selector Güncellemeleri

Trendyol sitesi değiştiyse:
1. `scraping/scrapers/trendyol_scraper.py` dosyasını açın
2. `SELECTORS` dictionary'sini güncelleyin
3. Browser developer tools ile yeni selector'ları bulun

## 📊 Performans

### Hız Karşılaştırması

- **API Modu** (varsayımsal): ~100-200ms/istek
- **Scraping Modu**: ~3-5 saniye/ürün
  - Browser başlatma: ~2 saniye
  - Sayfa yükleme: ~1-2 saniye
  - Data extraction: ~0.5-1 saniye

### Optimizasyonlar

1. **Cache Kullanımı**: İlk istek yavaş, sonrakiler cache'den hızlı
2. **Browser Reuse**: Singleton pattern ile browser yeniden kullanılır
3. **Batch Processing**: Birden fazla ürünü aynı browser session'da işle

## 🧪 Test

Backend testleri güncellenmesi gerekebilir:

```python
# Mock scraping service
@pytest.fixture
def mock_trendyol_service():
    service = Mock(spec=TrendyolScrapingService)
    service.search_products.return_value = [...]
    return service
```

## 📝 Notlar

- Scraping, API'ye göre daha yavaş ancak daha güvenilirdir
- Rate limiting bottan kaçınmak için kritiktir
- Cache kullanımı performans için önemlidir
- Production'da headless=True kullanın
- Development'ta headless=False ile debug yapabilirsiniz

## 🔐 Yasal Uyarı

Web scraping yaparken:
- robots.txt'ye uyun
- Rate limiting kullanın
- Site'ye zarar vermeyin
- Telif haklarına saygı gösterin
- Kullanım koşullarını okuyun
