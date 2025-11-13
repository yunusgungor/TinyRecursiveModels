# Implementation Plan

- [x] 1. Proje yapısını ve temel konfigürasyonu oluştur
  - `scraping/` klasör yapısını oluştur (config, scrapers, services, utils, logs)
  - Scraping için gerekli bağımlılıkları `requirements_scraping.txt` dosyasına ekle
  - Yapılandırma dosyası `config/scraping_config.yaml` oluştur
  - _Requirements: 6.1, 6.2_

- [x] 2. Configuration Manager ve Logger bileşenlerini implement et
  - [x] 2.1 ConfigurationManager sınıfını oluştur
    - YAML dosyasını okuma ve parse etme fonksiyonları yaz
    - Website, rate limit, Gemini ve output konfigürasyonlarını döndüren metodlar ekle
    - Test mode kontrolü ekle
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 2.2 ScrapingLogger sınıfını oluştur
    - Rotating file handler ile logging setup yap
    - Ana log ve error log dosyalarını yapılandır
    - Verbose mode desteği ekle
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

- [x] 3. Rate Limiter ve anti-bot mekanizmalarını implement et
  - [x] 3.1 RateLimiter sınıfını oluştur
    - Asenkron semaphore ile eşzamanlı istek kontrolü yap
    - Dakika başına istek limiti uygula
    - Rastgele bekleme süresi ekle
    - _Requirements: 2.2, 2.5_
  
  - [x] 3.2 Anti-bot stratejilerini ekle
    - User agent rotation fonksiyonu yaz
    - Random delay mekanizması ekle
    - Request timing tracker implement et
    - _Requirements: 2.1, 2.3_

- [x] 4. Data validation ve model sınıflarını oluştur
  - [x] 4.1 Pydantic data modellerini tanımla
    - RawProductData model sınıfını oluştur
    - EnhancedProductData model sınıfını oluştur
    - Validatorlar ekle (name, description, price kontrolü)
    - _Requirements: 1.3, 1.5_
  
  - [x] 4.2 DataValidator sınıfını implement et
    - Tek ürün validation fonksiyonu yaz
    - Batch validation fonksiyonu yaz
    - Duplicate removal fonksiyonu ekle
    - _Requirements: 4.4_

- [x] 5. Base Scraper abstract sınıfını oluştur
  - BaseScraper abstract class tanımla
  - Abstract metodları belirle (scrape_products, extract_product_details)
  - Ortak utility metodları ekle (wait_random_delay, setup_browser)
  - Playwright browser initialization ekle
  - _Requirements: 1.1, 2.1, 2.2_

- [x] 6. Çiçek Sepeti scraper'ını implement et
  - [x] 6.1 CicekSepetiScraper sınıfını oluştur
    - BaseScraper'dan türet
    - Site-specific selectors tanımla
    - _Requirements: 1.1_
  
  - [x] 6.2 Ürün listesi scraping fonksiyonunu yaz
    - Kategori sayfalarına navigate et
    - Ürün listelerini parse et
    - Pagination handling ekle
    - _Requirements: 1.1, 1.2_
  
  - [x] 6.3 Ürün detay extraction fonksiyonunu yaz
    - Ürün sayfasından name, price, description çıkar
    - Image URL, rating, stock status bilgilerini al
    - Error handling ekle
    - _Requirements: 1.3, 1.4_

- [x] 7. Hepsiburada scraper'ını implement et
  - [x] 7.1 HepsiburadaScraper sınıfını oluştur
    - BaseScraper'dan türet
    - Site-specific selectors tanımla
    - _Requirements: 1.1_
  
  - [x] 7.2 Ürün listesi ve detay extraction fonksiyonlarını yaz
    - Kategori navigation ve ürün listesi parse et
    - Ürün detaylarını çıkar (name, price, description, vb.)
    - Pagination ve error handling ekle
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 8. Trendyol scraper'ını implement et
  - [x] 8.1 TrendyolScraper sınıfını oluştur
    - BaseScraper'dan türet
    - Site-specific selectors tanımla
    - _Requirements: 1.1_
  
  - [x] 8.2 Ürün listesi ve detay extraction fonksiyonlarını yaz
    - Kategori navigation ve ürün listesi parse et
    - Ürün detaylarını çıkar (name, price, description, vb.)
    - Pagination ve error handling ekle
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 9. Scraping Orchestrator'ı implement et
  - [x] 9.1 ScrapingOrchestrator sınıfını oluştur
    - Scraper factory pattern ile scraper'ları initialize et
    - Rate limiter entegrasyonu yap
    - _Requirements: 1.1, 1.2_
  
  - [x] 9.2 Multi-website scraping fonksiyonunu yaz
    - Tüm enabled website'leri paralel scrape et
    - Progress tracking ekle
    - Error handling ve recovery mekanizması ekle
    - _Requirements: 1.4, 5.1, 5.2_
  
  - [x] 9.3 CAPTCHA detection ve handling ekle
    - CAPTCHA tespit mekanizması yaz
    - İşlemi duraklat ve kullanıcıyı bilgilendir
    - _Requirements: 2.4_

- [x] 10. Gemini Enhancement Service'i implement et
  - [x] 10.1 GeminiEnhancementService sınıfını oluştur
    - Gemini API client initialize et
    - API key environment variable'dan al
    - Request counter ve daily limit kontrolü ekle
    - _Requirements: 3.1, 3.5_
  
  - [x] 10.2 Product enhancement fonksiyonunu yaz
    - Prompt builder fonksiyonu yaz
    - Gemini API'ye asenkron istek gönder
    - JSON response parser ekle
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 10.3 Retry ve error handling ekle
    - 3 deneme ile retry mekanizması ekle
    - Exponential backoff uygula
    - Fallback kategorization ekle
    - _Requirements: 3.4_
  
  - [x] 10.4 Batch processing fonksiyonunu yaz
    - Ürünleri batch'lere böl
    - Paralel enhancement işle
    - Progress logging ekle
    - _Requirements: 3.1, 5.1_

- [x] 11. Dataset Generator'ı implement et
  - [x] 11.1 DatasetGenerator sınıfını oluştur
    - Output path konfigürasyonu ekle
    - _Requirements: 4.1, 4.5_
  
  - [x] 11.2 Data merging fonksiyonunu yaz
    - Raw product data ile AI enhancement'ları birleştir
    - Missing data handling ekle
    - _Requirements: 4.1_
  
  - [x] 11.3 Gift catalog format conversion fonksiyonunu yaz
    - Mevcut gift catalog formatına dönüştür
    - Unique ID generation ekle
    - Field mapping yap
    - _Requirements: 4.1, 4.2_
  
  - [x] 11.4 Metadata generation fonksiyonunu yaz
    - Kategori dağılımı hesapla
    - Fiyat istatistikleri çıkar
    - Source bilgilerini topla
    - _Requirements: 4.3_
  
  - [x] 11.5 Dataset save fonksiyonunu yaz
    - JSON formatında kaydet
    - UTF-8 encoding kullan
    - Pretty print ile okunabilir format
    - _Requirements: 4.5_

- [x] 12. Ana pipeline script'ini oluştur
  - [x] 12.1 Main pipeline orchestration script'i yaz
    - Tüm bileşenleri initialize et
    - Pipeline akışını koordine et (scrape -> validate -> enhance -> generate)
    - _Requirements: 1.1, 1.2_
  
  - [x] 12.2 CLI interface ekle
    - argparse ile command line arguments ekle
    - Test mode, website selection, verbose mode seçenekleri ekle
    - _Requirements: 6.5_
  
  - [x] 12.3 Progress reporting ekle
    - Her 100 üründe progress log yaz
    - Toplam/başarılı/başarısız istatistikleri göster
    - Geçen süre hesapla
    - _Requirements: 5.1, 5.2_
  
  - [x] 12.4 Final summary report oluştur
    - Scraping özeti oluştur
    - Başarı oranları hesapla
    - Kaydedilen dosya yollarını göster
    - _Requirements: 5.4_

- [x] 13. Error handling ve logging entegrasyonu
  - Tüm bileşenlerde try-catch blokları ekle
  - Anlamlı error mesajları yaz
  - Error'ları log dosyasına kaydet
  - Kritik hatalarda graceful shutdown yap
  - _Requirements: 1.4, 5.3_

- [x] 14. Raw ve processed data kaydetme fonksiyonlarını ekle
  - Raw scraped data'yı JSON olarak kaydet
  - Processed data'yı intermediate format'ta kaydet
  - Backup mekanizması ekle
  - _Requirements: 1.5_

- [x] 15. Test suite oluştur
  - [x] 15.1 Unit testler yaz
    - ConfigurationManager testleri
    - DataValidator testleri
    - RateLimiter testleri
    - DatasetGenerator testleri
  
  - [x] 15.2 Integration testler yaz
    - Scraper + Validator integration testi
    - Validator + Gemini Service integration testi
    - End-to-end pipeline testi (test mode ile)
  
  - [x] 15.3 Mock testler yaz
    - Gemini API mock testleri
    - Playwright browser mock testleri

- [x] 16. Dokümantasyon oluştur
  - [x] 16.1 README.md dosyası yaz
    - Kurulum talimatları
    - Kullanım örnekleri
    - Konfigürasyon açıklamaları
  
  - [x] 16.2 API dokümantasyonu yaz
    - Her sınıf için docstring'ler ekle
    - Fonksiyon parametrelerini dokümante et
    - Örnek kullanımlar ekle
