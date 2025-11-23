# Requirements Document

## Introduction

Bu doküman, eğitilmiş TinyRecursiveModels (TRM) modelini kullanarak Trendyol üzerinden gerçek zamanlı ürün önerileri sunan bir web uygulamasının gereksinimlerini tanımlar. Sistem, kullanıcı profillerine göre kişiselleştirilmiş hediye önerileri sunacak ve altı farklı analiz aracı (price comparison, inventory check, review analysis, trend analysis, budget optimizer, gift recommendation) kullanarak kapsamlı ürün araştırması yapacaktır.

## Glossary

- **TRM (Tiny Recursive Model)**: Recursive reasoning yapabilen, tool kullanımı entegre edilmiş derin öğrenme modeli
- **Web Client**: React ve TypeScript ile geliştirilmiş kullanıcı arayüzü uygulaması
- **Backend API**: Model inference ve tool orchestration yapan sunucu tarafı uygulama
- **Tool**: Ürün analizi için kullanılan özelleşmiş fonksiyon (price comparison, review analysis, vb.)
- **User Profile**: Kullanıcının yaş, hobiler, bütçe, ilişki durumu gibi özelliklerini içeren veri yapısı
- **Gift Item**: Önerilebilir ürün bilgilerini içeren veri yapısı
- **Recommendation Engine**: TRM modelini kullanarak ürün önerileri üreten sistem bileşeni
- **Trendyol API**: Trendyol e-ticaret platformunun ürün verilerine erişim sağlayan arayüz
- **Real-time Inference**: Kullanıcı isteği anında model çalıştırma ve sonuç üretme
- **Tool Orchestration**: Birden fazla aracın koordineli şekilde çalıştırılması

## Requirements

### Requirement 1: Kullanıcı Profili Yönetimi

**User Story:** Bir kullanıcı olarak, hediye alacağım kişinin özelliklerini sisteme girebilmek istiyorum, böylece kişiselleştirilmiş öneriler alabilirim.

#### Acceptance Criteria

1. WHEN kullanıcı profil oluşturma sayfasını açtığında THEN Web Client yaş, hobiler, bütçe, ilişki durumu, özel gün ve kişilik özellikleri için form alanları göstermelidir
2. WHEN kullanıcı yaş alanına değer girdiğinde THEN Web Client 18 ile 100 arasında sayısal değer kabul etmelidir
3. WHEN kullanıcı hobi seçimi yaptığında THEN Web Client çoklu seçim yapılabilir hobi listesi sunmalıdır
4. WHEN kullanıcı bütçe girdiğinde THEN Web Client pozitif sayısal değer kabul etmeli ve Türk Lirası formatında göstermelidir
5. WHEN kullanıcı profil formunu tamamladığında THEN Web Client tüm zorunlu alanların doldurulduğunu doğrulamalıdır
6. WHEN kullanıcı profili kaydedildiğinde THEN Backend API profil verisini JSON formatında saklamalıdır

### Requirement 2: Model Entegrasyonu ve Inference

**User Story:** Bir sistem yöneticisi olarak, eğitilmiş TRM modelinin web uygulamasına entegre edilmesini istiyorum, böylece gerçek zamanlı öneriler üretilebilsin.

#### Acceptance Criteria

1. WHEN Backend API başlatıldığında THEN sistem checkpoint dosyasından eğitilmiş model ağırlıklarını yüklemelidir
2. WHEN model yükleme işlemi başarısız olduğunda THEN Backend API anlamlı hata mesajı döndürmeli ve uygulamayı başlatmamalıdır
3. WHEN kullanıcı öneri talebi gönderdiğinde THEN Backend API kullanıcı profilini model input formatına dönüştürmelidir
4. WHEN model inference çalıştırıldığında THEN Backend API GPU kullanılabilir ise GPU üzerinde, değilse CPU üzerinde çalıştırmalıdır
5. WHEN model çıktısı alındığında THEN Backend API model skorlarını öneri listesine dönüştürmelidir
6. WHEN inference süresi 5 saniyeyi aştığında THEN Backend API timeout hatası döndürmelidir

### Requirement 3: Tool Orchestration Sistemi

**User Story:** Bir kullanıcı olarak, sistemin ürünler hakkında kapsamlı analiz yapmasını istiyorum, böylece en iyi seçeneği bulabileyim.

#### Acceptance Criteria

1. WHEN model tool seçimi yaptığında THEN Backend API seçilen tool'ları sırayla çalıştırmalıdır
2. WHEN price comparison tool çağrıldığında THEN sistem Trendyol API'den ürün fiyat bilgilerini almalıdır
3. WHEN inventory check tool çağrıldığında THEN sistem ürün stok durumunu kontrol etmelidir
4. WHEN review analysis tool çağrıldığında THEN sistem ürün yorumlarını analiz etmeli ve sentiment skoru hesaplamalıdır
5. WHEN trend analysis tool çağrıldığında THEN sistem kategori trendlerini ve popülerlik skorlarını hesaplamalıdır
6. WHEN budget optimizer tool çağrıldığında THEN sistem bütçe dağılımı önerileri üretmelidir
7. WHEN herhangi bir tool 3 saniyeden uzun sürdüğünde THEN Backend API tool çalıştırmasını iptal etmeli ve sonraki tool'a geçmelidir
8. WHEN tool çalıştırması başarısız olduğunda THEN Backend API hata loglamalı ancak diğer tool'ları çalıştırmaya devam etmelidir

### Requirement 4: Trendyol API Entegrasyonu

**User Story:** Bir sistem geliştiricisi olarak, Trendyol'dan gerçek zamanlı ürün verisi çekebilmek istiyorum, böylece güncel öneriler sunabilirim.

#### Acceptance Criteria

1. WHEN sistem ürün araması yaptığında THEN Backend API Trendyol API'ye kategori ve anahtar kelime parametreleri ile istek göndermelidir
2. WHEN Trendyol API yanıt döndüğünde THEN Backend API ürün verilerini Gift Item formatına dönüştürmelidir
3. WHEN API rate limit aşıldığında THEN Backend API istekleri kuyruğa almalı ve rate limit sıfırlanana kadar beklemelidir
4. WHEN API yanıtı hatalı olduğunda THEN Backend API önbelleğe alınmış verileri kullanmalıdır
5. WHEN ürün görselleri alındığında THEN Backend API görsel URL'lerini doğrulamalı ve geçersiz URL'leri filtrelemelidir
6. WHEN ürün fiyatları alındığında THEN Backend API fiyatları Türk Lirası formatında normalize etmelidir

### Requirement 5: Öneri Sonuçlarının Görselleştirilmesi

**User Story:** Bir kullanıcı olarak, önerilen ürünleri görsel ve detaylı şekilde görmek istiyorum, böylece karar vermem kolaylaşsın.

#### Acceptance Criteria

1. WHEN öneri sonuçları alındığında THEN Web Client her ürün için kart bileşeni göstermelidir
2. WHEN ürün kartı görüntülendiğinde THEN Web Client ürün görseli, isim, fiyat, rating ve kategori bilgilerini göstermelidir
3. WHEN kullanıcı ürün kartına tıkladığında THEN Web Client detaylı ürün bilgileri modal penceresi açmalıdır
4. WHEN detay modal açıldığında THEN Web Client tool analiz sonuçlarını (fiyat karşılaştırma, yorum analizi, vb.) göstermelidir
5. WHEN kullanıcı "Trendyol'da Gör" butonuna tıkladığında THEN Web Client yeni sekmede Trendyol ürün sayfasını açmalıdır
6. WHEN öneri güven skoru düşük olduğunda THEN Web Client kullanıcıyı uyarı mesajı ile bilgilendirmelidir

### Requirement 6: Performans ve Önbellekleme

**User Story:** Bir sistem yöneticisi olarak, uygulamanın hızlı yanıt vermesini istiyorum, böylece kullanıcı deneyimi olumsuz etkilenmesin.

#### Acceptance Criteria

1. WHEN aynı kullanıcı profili için tekrar istek geldiğinde THEN Backend API önbellekten sonuç döndürmelidir
2. WHEN önbellek verisi 1 saatten eski olduğunda THEN Backend API yeni inference çalıştırmalıdır
3. WHEN Trendyol API verisi önbelleğe alındığında THEN Backend API 30 dakika TTL (Time To Live) uygulamalıdır
4. WHEN eşzamanlı birden fazla istek geldiğinde THEN Backend API istekleri kuyruğa almalı ve sırayla işlemelidir
5. WHEN sistem yükü yüksek olduğunda THEN Backend API yanıt süresini kullanıcıya bildirmelidir
6. WHEN önbellek boyutu 500 MB'ı aştığında THEN Backend API en eski kayıtları silmelidir

### Requirement 7: Hata Yönetimi ve Logging

**User Story:** Bir sistem yöneticisi olarak, hataların loglanmasını ve kullanıcıya anlamlı mesajlar gösterilmesini istiyorum, böylece sorunları hızlıca çözebilirim.

#### Acceptance Criteria

1. WHEN herhangi bir hata oluştuğunda THEN Backend API hatayı timestamp, hata tipi ve stack trace ile loglamalıdır
2. WHEN model inference hatası oluştuğunda THEN Backend API kullanıcıya "Model şu anda kullanılamıyor" mesajı göstermelidir
3. WHEN Trendyol API erişilemez olduğunda THEN Backend API kullanıcıya "Ürün verileri şu anda alınamıyor" mesajı göstermelidir
4. WHEN validation hatası oluştuğunda THEN Backend API hangi alanın hatalı olduğunu belirten mesaj döndürmelidir
5. WHEN kritik hata oluştuğunda THEN Backend API sistem yöneticisine email bildirimi göndermelidir
6. WHEN log dosyası 100 MB'ı aştığında THEN Backend API log rotation yapmalıdır

### Requirement 8: Kullanıcı Arayüzü ve Deneyim

**User Story:** Bir kullanıcı olarak, uygulamayı kolayca kullanabilmek istiyorum, böylece hediye aramam hızlı ve keyifli olsun.

#### Acceptance Criteria

1. WHEN kullanıcı ana sayfayı açtığında THEN Web Client responsive tasarım ile mobil ve masaüstünde düzgün görünmelidir
2. WHEN kullanıcı form doldururken THEN Web Client gerçek zamanlı validasyon feedback'i göstermelidir
3. WHEN öneri yüklenirken THEN Web Client loading animasyonu ve ilerleme göstergesi göstermelidir
4. WHEN kullanıcı önceki aramaları görmek istediğinde THEN Web Client arama geçmişi listesi sunmalıdır
5. WHEN kullanıcı favorilere ekleme yaptığında THEN Web Client ürünü local storage'a kaydetmelidir
6. WHEN kullanıcı karanlık mod seçtiğinde THEN Web Client tüm bileşenleri karanlık tema ile göstermelidir

### Requirement 9: Güvenlik ve Veri Koruma

**User Story:** Bir kullanıcı olarak, verilerimin güvenli şekilde işlenmesini istiyorum, böylece gizliliğim korunsun.

#### Acceptance Criteria

1. WHEN kullanıcı veri gönderdiğinde THEN Backend API HTTPS protokolü kullanmalıdır
2. WHEN API istekleri yapıldığında THEN Backend API rate limiting uygulamalıdır (kullanıcı başına dakikada 10 istek)
3. WHEN kullanıcı verisi saklandığında THEN Backend API kişisel bilgileri şifrelemeli
4. WHEN SQL injection girişimi tespit edildiğinde THEN Backend API isteği reddetmeli ve loglamalıdır
5. WHEN XSS saldırısı tespit edildiğinde THEN Web Client input'ları sanitize etmelidir
6. WHEN kullanıcı oturumu 30 dakika boyunca aktif olmadığında THEN sistem oturumu sonlandırmalıdır

### Requirement 10: Test Edilebilirlik ve Monitoring

**User Story:** Bir geliştirici olarak, sistemin sağlığını izleyebilmek istiyorum, böylece proaktif olarak sorunları tespit edebilirim.

#### Acceptance Criteria

1. WHEN sistem çalışırken THEN Backend API health check endpoint'i sunmalıdır
2. WHEN health check çağrıldığında THEN Backend API model durumu, API bağlantısı ve önbellek durumunu döndürmelidir
3. WHEN inference süresi ölçüldüğünde THEN Backend API metriği Prometheus formatında export etmelidir
4. WHEN tool kullanım istatistikleri toplandığında THEN Backend API hangi tool'ların ne sıklıkla kullanıldığını loglamalıdır
5. WHEN API yanıt süreleri 2 saniyeyi aştığında THEN Backend API uyarı metriği üretmelidir
6. WHEN sistem kaynakları izlendiğinde THEN Backend API CPU, memory ve GPU kullanımını raporlamalıdır
