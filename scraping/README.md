# Web Scraping Data Pipeline

E-ticaret sitelerinden ürün verilerini toplayarak hediye öneri modeli için yüksek kaliteli veri seti oluşturan otomatik scraping pipeline'ı.

## Özellikler

- 🌐 **Multi-Website Scraping**: Çiçek Sepeti, Hepsiburada ve Trendyol'dan veri toplama
- 🤖 **AI Enhancement**: Gemini API ile ürün verilerini zenginleştirme
- 🛡️ **Anti-Bot Protection**: Rate limiting, user agent rotation ve CAPTCHA detection
- ✅ **Data Validation**: Pydantic ile güçlü veri doğrulama
- 📊 **Dataset Generation**: Model eğitimi için hazır veri seti oluşturma
- 🎯 **Dynamic User Scenario Generation**: Gerçek scraped veriden otomatik kullanıcı senaryoları oluşturma

## Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Playwright Browser'ı Kurun

```bash
playwright install chromium
```

### 3. Environment Variables

`.env` dosyası oluşturun:

```bash
cp .env.example .env
```

Gemini API key'inizi ekleyin:

```
GEMINI_API_KEY=your_api_key_here
```

## Kullanım

### 1. Tam Pipeline (Önerilen)

Tek komutla hem gift catalog hem user scenarios oluşturur:

```bash
# Temel kullanım (scraping + enhancement + scenarios)
python scripts/scraping.py

# Test modu (hızlı test)
python scripts/scraping.py --test

# Belirli bir siteden scraping
python scripts/scraping.py --website trendyol

# Verbose logging
python scripts/scraping.py --verbose
```

Pipeline otomatik olarak şunları yapar:
1. Web scraping
2. Veri validasyonu
3. Gemini ile enhancement
4. Gift catalog oluşturma
5. **User scenarios oluşturma** (gerçek veriden dinamik)

### 2. Sadece User Scenarios Test

Mevcut gift catalog ile scenario generation'ı test etmek için:

```bash
python scraping/scripts/test_scenario_generator.py
```

## Konfigürasyon

`config/scraping_config.yaml` dosyasını düzenleyerek ayarları özelleştirebilirsiniz:

### Website Ayarları

```yaml
scraping:
  websites:
    - name: "ciceksepeti"
      enabled: true
      max_products: 500
      categories:
        - "hediye"
        - "cicek"
```

### Rate Limiting

```yaml
rate_limit:
  requests_per_minute: 20
  delay_between_requests: [2, 5]
  max_concurrent_requests: 10
```

### Gemini API

```yaml
gemini:
  model: "gemini-1.5-flash"
  max_requests_per_day: 1000
  retry_attempts: 3
```

### Output Ayarları

```yaml
output:
  final_dataset_path: "data/scraped_gift_catalog.json"
  user_scenarios_path: "data/user_scenarios.json"
  num_user_scenarios: 100  # Oluşturulacak senaryo sayısı
```

## Proje Yapısı

```
scraping/
├── config/              # Konfigürasyon yönetimi
├── scrapers/            # Web scraper'lar
│   ├── base_scraper.py
│   ├── ciceksepeti_scraper.py
│   ├── hepsiburada_scraper.py
│   ├── trendyol_scraper.py
│   └── orchestrator.py
├── services/            # Servisler
│   ├── gemini_service.py
│   └── dataset_generator.py
└── utils/               # Yardımcı araçlar
    ├── models.py
    ├── validator.py
    ├── rate_limiter.py
    ├── anti_bot.py
    └── logger.py
```

## Pipeline Aşamaları

1. **Scraping**: Web sitelerinden ürün verilerini toplama
2. **Validation**: Verileri doğrulama ve temizleme
3. **Enhancement**: Gemini API ile verileri zenginleştirme
4. **Dataset Generation**: Final gift catalog'u oluşturma
5. **User Scenario Generation**: Gerçek veriden dinamik senaryolar oluşturma
   - Gerçek kategorileri kullanır
   - Gerçek tag'leri hobi/tercih olarak kullanır
   - Gerçek occasions'ları kullanır
   - Gerçek fiyat aralıklarını kullanır

## Output

Pipeline çalıştırıldığında şu dosyalar oluşturulur:

### Gift Catalog
- `data/scraped_gift_catalog.json` - Final gift catalog veri seti
- `data/scraped_raw/` - Ham scraping verileri
- `data/scraped_processed/` - İşlenmiş veriler

### User Scenarios
- `data/user_scenarios.json` - Kullanıcı senaryoları veri seti

### Logs
- `logs/scraping.log` - Ana log dosyası
- `logs/scraping_errors.log` - Hata logları
- `logs/user_scenario_generation.log` - Senaryo oluşturma logları

## Troubleshooting

### CAPTCHA Detected

Eğer CAPTCHA ile karşılaşırsanız:
- Rate limit ayarlarını düşürün
- Delay süresini artırın
- Daha sonra tekrar deneyin

### API Limit Exceeded

Gemini API limiti aşıldıysa:
- `max_requests_per_day` ayarını kontrol edin
- Ertesi gün tekrar deneyin
- Veya API key'inizi upgrade edin

### Selector Not Found

Web sitesi yapısı değiştiyse:
- İlgili scraper dosyasındaki `SELECTORS` dictionary'sini güncelleyin
- Browser'da inspect ederek yeni selector'ları bulun

## Lisans

MIT License

## Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır!
