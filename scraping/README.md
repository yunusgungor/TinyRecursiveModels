# Web Scraping Data Pipeline

E-ticaret sitelerinden ürün verilerini toplayarak hediye öneri modeli için yüksek kaliteli veri seti oluşturan otomatik scraping pipeline'ı.

## Özellikler

- 🌐 **Multi-Website Scraping**: Çiçek Sepeti, Hepsiburada ve Trendyol'dan veri toplama
- 🤖 **AI Enhancement**: Gemini API ile ürün verilerini zenginleştirme
- 🛡️ **Anti-Bot Protection**: Rate limiting, user agent rotation ve CAPTCHA detection
- ✅ **Data Validation**: Pydantic ile güçlü veri doğrulama
- 📊 **Dataset Generation**: Model eğitimi için hazır veri seti oluşturma
- 🎯 **User Scenario Generation**: Gift catalog'dan otomatik kullanıcı senaryoları oluşturma

## Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements_scraping.txt
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

### 1. Gift Catalog Oluşturma

```bash
# Temel kullanım
python scripts/run_scraping_pipeline.py

# Test modu (hızlı test)
python scripts/run_scraping_pipeline.py --test

# Belirli bir siteden scraping
python scripts/run_scraping_pipeline.py --website ciceksepeti

# Verbose logging
python scripts/run_scraping_pipeline.py --verbose
```

### 2. User Scenarios Oluşturma

Gift catalog oluşturduktan sonra kullanıcı senaryolarını oluşturun:

```bash
# 100 senaryo oluştur (varsayılan)
python scraping/scripts/generate_user_scenarios.py

# Özel sayıda senaryo
python scraping/scripts/generate_user_scenarios.py 200

# Gemini API ile (daha gerçekçi senaryolar)
export GEMINI_API_KEY="your-api-key"
python scraping/scripts/generate_user_scenarios.py 100
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
4. **Generation**: Final veri setini oluşturma

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

## Test

```bash
# Unit testleri çalıştır
pytest tests/

# Belirli bir test dosyası
pytest tests/test_validator.py

# Coverage ile
pytest --cov=scraping tests/
```

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
