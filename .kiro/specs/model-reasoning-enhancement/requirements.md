# Requirements Document

## Introduction

Bu doküman, Trendyol Gift Recommendation sisteminde modelin reasoning süreçlerinin (düşünme adımları, tool seçim mantığı, kategori eşleştirme açıklamaları, attention weights) backend'de yakalanması ve kullanıcıya sunulması için gereksinimleri tanımlar. Mevcut sistemde model temel tool sonuçlarını ve basit statik reasoning'i döndürmektedir, ancak modelin gerçek karar verme süreçleri (neden bu tool seçildi, kategoriler nasıl eşleştirildi, model neye odaklandı) eksiktir.

## Glossary

- **Model**: IntegratedEnhancedTRM - Eğitilmiş derin öğrenme modeli
- **Reasoning Trace**: Modelin karar verme sürecinin adım adım kaydı
- **Tool Selection**: Modelin hangi toolları (price_comparison, review_analysis, vb.) kullanacağına karar verme süreci
- **Category Matching**: Kullanıcı profiline göre hediye kategorilerinin eşleştirilmesi
- **Attention Weights**: Modelin hangi özelliklere (hobi, bütçe, yaş, vb.) ne kadar önem verdiğini gösteren ağırlıklar
- **Confidence Score**: Modelin bir öneriye olan güven skoru (0.0-1.0 arası)
- **Backend**: FastAPI tabanlı Python backend servisi
- **Model Inference Service**: Backend'de model çıkarımını yöneten servis
- **Gift Recommendation**: Kullanıcıya önerilen hediye objesi
- **User Profile**: Kullanıcının hobi, yaş, bütçe, occasion gibi özelliklerini içeren profil

## Requirements

### Requirement 1

**User Story:** Bir geliştirici olarak, modelin hangi toolları neden seçtiğini görmek istiyorum, böylece model davranışını anlayabilir ve debug edebilirim.

#### Acceptance Criteria

1. WHEN model bir inference yapar THEN the Model SHALL her seçilen tool için seçim nedenini, confidence skorunu ve öncelik sırasını içeren detaylı bilgi üretir
2. WHEN tool selection reasoning oluşturulur THEN the Model SHALL kullanıcı profilindeki hangi özelliklerin (budget constraint, quality preference, vb.) tool seçimini etkilediğini açıklar
3. WHEN birden fazla tool seçilir THEN the Model SHALL toolların öncelik sırasını ve bu sıralamayı etkileyen faktörleri belirtir
4. WHEN tool selection confidence düşükse THEN the Model SHALL düşük confidence nedenini açıklar
5. WHEN tool selection sonuçları API response'a eklenir THEN the Backend SHALL tool selection reasoning'i yapılandırılmış JSON formatında döner

### Requirement 2

**User Story:** Bir kullanıcı olarak, modelin hediye kategorilerini profilime nasıl eşleştirdiğini görmek istiyorum, böylece önerilerin mantığını anlayabilirim.

#### Acceptance Criteria

1. WHEN model kategori eşleştirmesi yapar THEN the Model SHALL her kategori için eşleştirme skorunu ve bu skoru etkileyen faktörleri hesaplar
2. WHEN kategori skoru yüksekse THEN the Model SHALL hangi kullanıcı özelliklerinin (hobi, yaş, occasion) bu kategoriyle güçlü eşleştiğini açıklar
3. WHEN kategori skoru düşükse THEN the Model SHALL neden düşük olduğunu (örn: yaş uyumsuzluğu, hobi eşleşmemesi) belirtir
4. WHEN top kategoriler belirlenir THEN the Model SHALL en az top 3 kategoriyi skorlarıyla birlikte döner
5. WHEN kategori reasoning API'ye eklenir THEN the Backend SHALL her kategori için skor ve açıklama içeren yapılandırılmış veri döner

### Requirement 3

**User Story:** Bir kullanıcı olarak, modelin bana neden bu hediyeyi önerdiğini detaylı ve dinamik açıklamalarla görmek istiyorum, böylece öneriye güvenebilirim.

#### Acceptance Criteria

1. WHEN model bir hediye önerir THEN the Backend SHALL kullanıcı profili ve model output'una dayalı dinamik reasoning oluşturur
2. WHEN hobi eşleşmesi varsa THEN the Backend SHALL hangi hobilerin eşleştiğini ve eşleşme derecesini açıklar
3. WHEN bütçe optimizasyonu yapılırsa THEN the Backend SHALL hediye fiyatının bütçenin yüzde kaçını kullandığını ve value assessment'ı belirtir
4. WHEN tool insights mevcutsa THEN the Backend SHALL her tool'un bulgularını (rating, trend, availability) reasoning'e entegre eder
5. WHEN yaş uygunluğu kontrolü yapılırsa THEN the Backend SHALL hediyenin kullanıcının yaşına uygun olup olmadığını açıklar
6. WHEN statik şablon reasoning kullanılır THEN the Backend SHALL bunu context-aware dinamik reasoning ile değiştirir

### Requirement 4

**User Story:** Bir geliştirici olarak, modelin hangi özelliklere ne kadar önem verdiğini (attention weights) görmek istiyorum, böylece model davranışını analiz edebilirim.

#### Acceptance Criteria

1. WHEN model inference yapar THEN the Model SHALL user features için attention weights hesaplar
2. WHEN user features attention weights hesaplanır THEN the Model SHALL hobbies, budget, age, occasion gibi özelliklerin ağırlıklarını döner
3. WHEN gift features attention weights hesaplanır THEN the Model SHALL category, price, rating gibi özelliklerin ağırlıklarını döner
4. WHEN attention weights normalize edilir THEN the Model SHALL tüm weights toplamının 1.0 olmasını sağlar
5. WHEN attention weights API'ye eklenir THEN the Backend SHALL weights'i görselleştirilebilir formatta (örn: yüzde değerleri) döner

### Requirement 5

**User Story:** Bir geliştirici olarak, modelin düşünme sürecini adım adım görmek istiyorum, böylece model pipeline'ını anlayabilir ve optimize edebilirim.

#### Acceptance Criteria

1. WHEN model inference başlar THEN the Model SHALL her major step için (encode, match, select, execute, rank) bir thinking step kaydı oluşturur
2. WHEN bir thinking step tamamlanır THEN the Model SHALL step numarası, action adı, sonuç ve insight içeren kayıt üretir
3. WHEN user encoding tamamlanır THEN the Model SHALL encoding sonucunun özetini (örn: "Strong cooking interest detected") ekler
4. WHEN tool execution tamamlanır THEN the Model SHALL tool sonuçlarının özetini (örn: "Found 15 items in budget") ekler
5. WHEN gift ranking tamamlanır THEN the Model SHALL ranking kriterlerini ve top seçimleri açıklar
6. WHEN thinking steps API'ye eklenir THEN the Backend SHALL kronolojik sırayla tüm steps'i döner

### Requirement 6

**User Story:** Bir kullanıcı olarak, modelin confidence skorunun nedenini anlamak istiyorum, böylece öneriye ne kadar güvenebileceğimi bilebilirim.

#### Acceptance Criteria

1. WHEN model bir confidence score üretir THEN the Model SHALL bu skorun nasıl hesaplandığını açıklar
2. WHEN confidence yüksekse THEN the Model SHALL hangi faktörlerin (güçlü kategori eşleşmesi, yüksek rating, vb.) skoru artırdığını belirtir
3. WHEN confidence düşükse THEN the Model SHALL hangi faktörlerin (zayıf eşleşme, sınırlı veri, vb.) skoru düşürdüğünü belirtir
4. WHEN confidence explanation oluşturulur THEN the Backend SHALL açıklamayı kullanıcı dostu dilde sunar
5. WHEN confidence threshold'ları tanımlanır THEN the Backend SHALL yüksek (>0.8), orta (0.5-0.8) ve düşük (<0.5) confidence için farklı açıklamalar üretir

### Requirement 7

**User Story:** Bir sistem yöneticisi olarak, reasoning enhancement'ın performans üzerinde minimal etkisi olmasını istiyorum, böylece sistem hızı korunur.

#### Acceptance Criteria

1. WHEN reasoning trace oluşturulur THEN the Model SHALL inference süresine maksimum %10 ek yük getirir
2. WHEN attention weights hesaplanır THEN the Model SHALL mevcut forward pass'ten yararlanır ve ekstra forward pass gerektirmez
3. WHEN reasoning data serialize edilir THEN the Backend SHALL efficient JSON serialization kullanır
4. WHEN büyük reasoning trace oluşur THEN the Backend SHALL optional truncation veya summarization sağlar
5. WHEN reasoning enhancement devre dışı bırakılmak istenirse THEN the Backend SHALL feature flag ile reasoning generation'ı kapatma imkanı sunar

### Requirement 8

**User Story:** Bir frontend geliştirici olarak, reasoning bilgilerini yapılandırılmış ve tutarlı formatta almak istiyorum, böylece UI'da kolayca gösterebilirim.

#### Acceptance Criteria

1. WHEN API response oluşturulur THEN the Backend SHALL reasoning bilgilerini tutarlı JSON schema'ya göre yapılandırır
2. WHEN tool selection reasoning döner THEN the Backend SHALL her tool için name, reason, confidence, priority alanlarını içerir
3. WHEN category matching döner THEN the Backend SHALL her kategori için category_name, score, reasons listesi içerir
4. WHEN attention weights döner THEN the Backend SHALL user_features ve gift_features altında feature adı ve weight çiftleri içerir
5. WHEN thinking steps döner THEN the Backend SHALL her step için step_number, action, result, insight alanlarını içerir
6. WHEN API schema dokümante edilir THEN the Backend SHALL OpenAPI spec'e reasoning response modellerini ekler

### Requirement 9

**User Story:** Bir test engineer olarak, reasoning generation'ın doğruluğunu test edebilmek istiyorum, böylece kalite güvencesi sağlayabilirim.

#### Acceptance Criteria

1. WHEN reasoning generation test edilir THEN the Backend SHALL her reasoning component için unit test içerir
2. WHEN tool selection reasoning test edilir THEN the Backend SHALL farklı user profile'lar için doğru reasoning üretildiğini doğrular
3. WHEN category matching reasoning test edilir THEN the Backend SHALL kategori skorlarının açıklamalarla tutarlı olduğunu kontrol eder
4. WHEN attention weights test edilir THEN the Backend SHALL weights toplamının 1.0 olduğunu ve negatif değer olmadığını doğrular
5. WHEN thinking steps test edilir THEN the Backend SHALL tüm major steps'in kaydedildiğini ve kronolojik sırada olduğunu kontrol eder
6. WHEN property-based testing uygulanır THEN the Backend SHALL reasoning generation'ın farklı input kombinasyonlarında tutarlı davrandığını doğrular

### Requirement 10

**User Story:** Bir kullanıcı olarak, reasoning bilgilerini isteğe bağlı olarak almak istiyorum, böylece basit kullanımda gereksiz veri almam.

#### Acceptance Criteria

1. WHEN API request yapılır THEN the Backend SHALL optional include_reasoning query parametresi kabul eder
2. WHEN include_reasoning=false ise THEN the Backend SHALL sadece temel recommendation bilgilerini döner
3. WHEN include_reasoning=true ise THEN the Backend SHALL tam reasoning trace'i döner
4. WHEN include_reasoning parametresi belirtilmezse THEN the Backend SHALL default olarak basic reasoning döner
5. WHEN reasoning level kontrolü yapılır THEN the Backend SHALL basic, detailed, full gibi farklı reasoning seviyeleri destekler
