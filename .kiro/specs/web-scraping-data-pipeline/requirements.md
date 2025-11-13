# Requirements Document

## Introduction

Bu özellik, Türkiye'deki önde gelen e-ticaret platformlarından (Çiçek Sepeti, Hepsiburada, Trendyol) gerçek ürün verilerini otomatik olarak toplayarak, hediye öneri modelinin eğitimi için yüksek kaliteli bir veri seti oluşturmayı amaçlamaktadır. Sistem, Gemini API ve headless browser teknolojilerini kullanarak ürün bilgilerini, kategorileri, fiyatları ve diğer ilgili meta verileri çıkaracaktır.

## Glossary

- **Scraping System**: Web sitelerinden veri toplayan otomatik sistem
- **Headless Browser**: Grafik arayüzü olmadan çalışan web tarayıcısı (örn: Playwright, Puppeteer)
- **Gemini API**: Google'ın yapay zeka API'si, veri işleme ve zenginleştirme için kullanılır
- **Data Pipeline**: Veri toplama, işleme ve depolama süreçlerinin tamamı
- **Product Entity**: Ürün bilgilerini içeren veri yapısı (isim, fiyat, kategori, açıklama, vb.)
- **Rate Limiter**: İstek hızını kontrol eden mekanizma
- **Dataset Generator**: Toplanan verilerden eğitim veri seti oluşturan bileşen

## Requirements

### Requirement 1

**User Story:** Bir veri bilimcisi olarak, gerçek e-ticaret sitelerinden ürün verilerini otomatik olarak toplamak istiyorum, böylece modelimi gerçek dünya verileriyle eğitebilirim.

#### Acceptance Criteria

1. WHEN kullanıcı scraping işlemini başlattığında, THE Scraping System SHALL hedef web sitelerine bağlanıp ürün listelerini yükleyecektir
2. WHILE scraping işlemi devam ederken, THE Scraping System SHALL her web sitesi için ayrı ayrı ürün verilerini toplayacaktır
3. THE Scraping System SHALL her ürün için isim, fiyat, kategori, açıklama, resim URL'si ve stok durumu bilgilerini çıkaracaktır
4. IF bir web sitesine erişim başarısız olursa, THEN THE Scraping System SHALL hatayı loglayıp diğer sitelere devam edecektir
5. THE Scraping System SHALL toplanan verileri JSON formatında yapılandırılmış şekilde kaydedecektir

### Requirement 2

**User Story:** Bir sistem yöneticisi olarak, web sitelerinin anti-bot mekanizmalarını aşabilmek istiyorum, böylece veri toplama işlemi kesintisiz devam edebilir.

#### Acceptance Criteria

1. THE Scraping System SHALL headless browser teknolojisi kullanarak gerçek bir kullanıcı gibi davranacaktır
2. THE Scraping System SHALL her istek arasında 2-5 saniye rastgele bekleme süresi uygulayacaktır
3. THE Scraping System SHALL farklı user-agent başlıkları kullanarak istekleri çeşitlendirecektir
4. IF bir CAPTCHA veya bot kontrolü tespit edilirse, THEN THE Scraping System SHALL işlemi duraklatıp kullanıcıyı bilgilendirecektir
5. THE Scraping System SHALL maksimum 10 eşzamanlı istek limiti uygulayacaktır

### Requirement 3

**User Story:** Bir veri bilimcisi olarak, toplanan ham verilerin Gemini API ile zenginleştirilmesini istiyorum, böylece daha anlamlı ve kategorize edilmiş veriler elde edebilirim.

#### Acceptance Criteria

1. WHEN bir ürün verisi toplandığında, THE Data Pipeline SHALL ürün açıklamasını Gemini API'ye gönderecektir
2. THE Data Pipeline SHALL Gemini API'den ürün kategorisi, hedef kitle, uygun hediye senaryoları ve duygusal etiketler alacaktır
3. THE Data Pipeline SHALL Gemini API yanıtlarını Product Entity'ye entegre edecektir
4. IF Gemini API isteği başarısız olursa, THEN THE Data Pipeline SHALL 3 kez yeniden deneme yapacaktır
5. THE Data Pipeline SHALL Gemini API kullanımını günlük 1000 istek ile sınırlandıracaktır

### Requirement 4

**User Story:** Bir makine öğrenmesi mühendisi olarak, toplanan verilerin model eğitimi için uygun formatta kaydedilmesini istiyorum, böylece doğrudan eğitim sürecinde kullanabilirim.

#### Acceptance Criteria

1. THE Dataset Generator SHALL toplanan verileri mevcut gift catalog formatına dönüştürecektir
2. THE Dataset Generator SHALL her ürün için benzersiz bir ID oluşturacaktır
3. THE Dataset Generator SHALL verileri kategorilere göre gruplandırıp istatistiksel özet çıkaracaktır
4. THE Dataset Generator SHALL eksik veya geçersiz verileri filtreleyecektir
5. THE Dataset Generator SHALL son veri setini 'data/scraped_gift_catalog.json' dosyasına kaydedecektir

### Requirement 5

**User Story:** Bir sistem yöneticisi olarak, scraping işleminin ilerlemesini ve durumunu izlemek istiyorum, böylece sorunları hızlıca tespit edip müdahale edebilirim.

#### Acceptance Criteria

1. THE Scraping System SHALL her 100 üründe bir ilerleme durumunu konsola yazdıracaktır
2. THE Scraping System SHALL toplam işlenen ürün sayısı, başarılı/başarısız istek sayıları ve geçen süreyi raporlayacaktır
3. THE Scraping System SHALL tüm hataları timestamp ile birlikte 'logs/scraping_errors.log' dosyasına kaydedecektir
4. WHEN scraping işlemi tamamlandığında, THE Scraping System SHALL özet rapor oluşturup kullanıcıya sunacaktır
5. WHERE kullanıcı verbose mod seçerse, THE Scraping System SHALL detaylı debug logları üretecektir

### Requirement 6

**User Story:** Bir veri bilimcisi olarak, scraping işlemini yapılandırabilmek istiyorum, böylece hangi siteleri, kaç ürünü ve hangi kategorileri toplayacağımı kontrol edebilirim.

#### Acceptance Criteria

1. THE Scraping System SHALL YAML formatında yapılandırma dosyası okuyacaktır
2. THE Scraping System SHALL hedef web siteleri listesini yapılandırmadan alacaktır
3. THE Scraping System SHALL her site için maksimum ürün sayısı limitini uygulayacaktır
4. THE Scraping System SHALL belirli kategorileri filtreleme seçeneği sunacaktır
5. WHERE kullanıcı test modu seçerse, THE Scraping System SHALL sadece 10 ürün toplayacaktır
