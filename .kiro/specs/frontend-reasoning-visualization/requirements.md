# Requirements Document

## Introduction

Bu doküman, Trendyol Gift Recommendation sisteminde backend'den gelen model reasoning bilgilerinin (düşünme adımları, tool seçim mantığı, kategori eşleştirme açıklamaları, attention weights) frontend'de görselleştirilmesi ve kullanıcıya sunulması için gereksinimleri tanımlar. Backend'de model-reasoning-enhancement ile eklenen reasoning trace, tool selection reasoning, category matching, attention weights ve thinking steps bilgileri kullanıcı dostu bir arayüzde gösterilecektir.

## Glossary

- **Frontend**: React + TypeScript tabanlı kullanıcı arayüzü
- **Reasoning Trace**: Backend'den gelen modelin karar verme sürecinin detaylı kaydı
- **Tool Selection Visualization**: Hangi toolların neden seçildiğini gösteren görsel bileşen
- **Category Matching Visualization**: Kategori eşleştirme skorlarını ve nedenlerini gösteren görsel bileşen
- **Attention Weights Visualization**: Model attention ağırlıklarını gösteren görsel bileşen (bar chart, radar chart)
- **Thinking Steps Timeline**: Modelin düşünme adımlarını kronolojik olarak gösteren timeline bileşeni
- **Confidence Indicator**: Güven skorunu ve açıklamasını gösteren görsel gösterge
- **Gift Recommendation Card**: Hediye önerisini ve reasoning'ini gösteren kart bileşeni
- **Reasoning Panel**: Detaylı reasoning bilgilerini gösteren genişletilebilir panel
- **User Profile**: Kullanıcının hobi, yaş, bütçe, occasion gibi özelliklerini içeren profil

## Requirements

### Requirement 1

**User Story:** Bir kullanıcı olarak, hediye önerilerini reasoning açıklamalarıyla birlikte görmek istiyorum, böylece önerilerin mantığını anlayabilirim.

#### Acceptance Criteria

1. WHEN hediye önerileri yüklenir THEN the Frontend SHALL her hediye için reasoning açıklamalarını kart üzerinde gösterir
2. WHEN reasoning açıklamaları gösterilir THEN the Frontend SHALL hobi eşleşmesi, bütçe optimizasyonu, yaş uygunluğu gibi faktörleri ayrı ayrı vurgular
3. WHEN tool insights mevcutsa THEN the Frontend SHALL rating, trend, availability bilgilerini icon'larla görselleştirir
4. WHEN reasoning metni uzunsa THEN the Frontend SHALL "Daha fazla göster" butonu ile genişletilebilir alan sağlar
5. WHEN kullanıcı reasoning'e tıklarsa THEN the Frontend SHALL detaylı reasoning panel'ini açar

### Requirement 2

**User Story:** Bir kullanıcı olarak, modelin hangi toolları neden seçtiğini görsel olarak görmek istiyorum, böylece model kararlarını anlayabilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL tool selection bölümünü gösterir
2. WHEN tool selection gösterilir THEN the Frontend SHALL her tool için seçim durumu, confidence skoru ve öncelik sırasını gösterir
3. WHEN bir tool seçilmişse THEN the Frontend SHALL tool'u yeşil renk ve checkmark icon ile vurgular
4. WHEN bir tool seçilmemişse THEN the Frontend SHALL tool'u gri renk ile gösterir
5. WHEN tool confidence skoru düşükse THEN the Frontend SHALL düşük confidence nedenini tooltip ile gösterir
6. WHEN tool'a hover yapılırsa THEN the Frontend SHALL seçim nedenini ve etkileyen faktörleri tooltip'te gösterir

### Requirement 3

**User Story:** Bir kullanıcı olarak, kategori eşleştirme skorlarını ve nedenlerini görmek istiyorum, böylece hangi kategorilerin neden önerildiğini anlayabilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL category matching bölümünü gösterir
2. WHEN category matching gösterilir THEN the Frontend SHALL en az top 3 kategoriyi skorlarıyla birlikte gösterir
3. WHEN kategori skoru yüksekse THEN the Frontend SHALL kategoriyi yeşil progress bar ile gösterir
4. WHEN kategori skoru düşükse THEN the Frontend SHALL kategoriyi kırmızı progress bar ile gösterir
5. WHEN kategoriye tıklanırsa THEN the Frontend SHALL eşleştirme nedenlerini (hobi, yaş, occasion) liste halinde gösterir
6. WHEN kategori skorları gösterilir THEN the Frontend SHALL skorları yüzde değeri olarak formatlar

### Requirement 4

**User Story:** Bir kullanıcı olarak, modelin hangi özelliklere ne kadar önem verdiğini (attention weights) görsel olarak görmek istiyorum, böylece model davranışını anlayabilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL attention weights bölümünü gösterir
2. WHEN user features attention gösterilir THEN the Frontend SHALL hobbies, budget, age, occasion ağırlıklarını bar chart ile görselleştirir
3. WHEN gift features attention gösterilir THEN the Frontend SHALL category, price, rating ağırlıklarını bar chart ile görselleştirir
4. WHEN attention weights normalize edilmişse THEN the Frontend SHALL her weight'i yüzde değeri olarak gösterir
5. WHEN bar'a hover yapılırsa THEN the Frontend SHALL feature adını ve tam değerini tooltip'te gösterir
6. WHEN kullanıcı görselleştirme tipini değiştirmek isterse THEN the Frontend SHALL bar chart ve radar chart arasında geçiş imkanı sunar

### Requirement 5

**User Story:** Bir kullanıcı olarak, modelin düşünme sürecini adım adım görmek istiyorum, böylece model pipeline'ını anlayabilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL thinking steps timeline bölümünü gösterir
2. WHEN thinking steps gösterilir THEN the Frontend SHALL her step'i kronolojik sırayla timeline üzerinde gösterir
3. WHEN bir step gösterilir THEN the Frontend SHALL step numarası, action adı, sonuç ve insight bilgilerini içerir
4. WHEN step tamamlanmışsa THEN the Frontend SHALL step'i yeşil checkmark ile işaretler
5. WHEN step'e tıklanırsa THEN the Frontend SHALL step detaylarını genişletilmiş formatta gösterir
6. WHEN timeline uzunsa THEN the Frontend SHALL scroll edilebilir alan sağlar

### Requirement 6

**User Story:** Bir kullanıcı olarak, confidence skorunu ve nedenini görsel olarak görmek istiyorum, böylece öneriye ne kadar güvenebileceğimi bilebilirim.

#### Acceptance Criteria

1. WHEN hediye önerisi gösterilir THEN the Frontend SHALL confidence skorunu görsel gösterge ile gösterir
2. WHEN confidence yüksekse (>0.8) THEN the Frontend SHALL yeşil renk ve "Yüksek Güven" etiketi gösterir
3. WHEN confidence ortaysa (0.5-0.8) THEN the Frontend SHALL sarı renk ve "Orta Güven" etiketi gösterir
4. WHEN confidence düşükse (<0.5) THEN the Frontend SHALL kırmızı renk ve "Düşük Güven" etiketi gösterir
5. WHEN confidence göstergesine tıklanırsa THEN the Frontend SHALL confidence açıklamasını (positive ve negative faktörler) modal'da gösterir
6. WHEN confidence açıklaması gösterilir THEN the Frontend SHALL faktörleri kategorize ederek (positive/negative) listeler

### Requirement 7

**User Story:** Bir kullanıcı olarak, reasoning bilgilerini isteğe bağlı olarak görmek istiyorum, böylece basit kullanımda arayüz karmaşık olmaz.

#### Acceptance Criteria

1. WHEN hediye önerileri yüklenir THEN the Frontend SHALL default olarak basic reasoning gösterir
2. WHEN kullanıcı "Detaylı Analiz Göster" butonuna tıklarsa THEN the Frontend SHALL detaylı reasoning panel'ini açar
3. WHEN detaylı panel açıksa THEN the Frontend SHALL "Detaylı Analiz Gizle" butonu gösterir
4. WHEN kullanıcı panel'i kapatırsa THEN the Frontend SHALL sadece basic reasoning'e geri döner
5. WHEN kullanıcı ayarlardan reasoning seviyesini değiştirirse THEN the Frontend SHALL seçilen seviyeyi (basic/detailed/full) localStorage'a kaydeder
6. WHEN sayfa yenilenirse THEN the Frontend SHALL kaydedilmiş reasoning seviyesini yükler

### Requirement 8

**User Story:** Bir geliştirici olarak, reasoning bileşenlerinin yeniden kullanılabilir olmasını istiyorum, böylece farklı sayfalarda kolayca kullanabilirim.

#### Acceptance Criteria

1. WHEN reasoning bileşenleri oluşturulur THEN the Frontend SHALL her bileşeni ayrı React component olarak tasarlar
2. WHEN ToolSelectionCard bileşeni oluşturulur THEN the Frontend SHALL tool selection reasoning'i props olarak alır
3. WHEN CategoryMatchingChart bileşeni oluşturulur THEN the Frontend SHALL category matching data'yı props olarak alır
4. WHEN AttentionWeightsChart bileşeni oluşturulur THEN the Frontend SHALL attention weights data'yı props olarak alır
5. WHEN ThinkingStepsTimeline bileşeni oluşturulur THEN the Frontend SHALL thinking steps array'ini props olarak alır
6. WHEN bileşenler TypeScript ile yazılırsa THEN the Frontend SHALL her bileşen için type definitions sağlar

### Requirement 9

**User Story:** Bir kullanıcı olarak, reasoning bilgilerinin yüklenmesini beklerken loading state görmek istiyorum, böylece sistemin çalıştığını bilebilirim.

#### Acceptance Criteria

1. WHEN reasoning bilgileri yüklenirken THEN the Frontend SHALL skeleton loader veya spinner gösterir
2. WHEN API isteği başarısız olursa THEN the Frontend SHALL hata mesajını kullanıcı dostu şekilde gösterir
3. WHEN reasoning bilgileri yoksa THEN the Frontend SHALL "Reasoning bilgisi mevcut değil" mesajı gösterir
4. WHEN retry imkanı sunulursa THEN the Frontend SHALL "Tekrar Dene" butonu gösterir
5. WHEN reasoning yüklenirken kullanıcı başka işlem yaparsa THEN the Frontend SHALL yükleme işlemini iptal eder

### Requirement 10

**User Story:** Bir kullanıcı olarak, reasoning görselleştirmelerinin responsive olmasını istiyorum, böylece mobil cihazlarda da rahatça kullanabilirim.

#### Acceptance Criteria

1. WHEN sayfa mobil cihazda açılır THEN the Frontend SHALL reasoning bileşenlerini mobil layout'a uyarlar
2. WHEN ekran genişliği 768px'den küçükse THEN the Frontend SHALL chart'ları dikey layout'ta gösterir
3. WHEN detaylı panel mobilde açılırsa THEN the Frontend SHALL full-screen modal olarak gösterir
4. WHEN touch gesture kullanılırsa THEN the Frontend SHALL swipe ile panel kapatma imkanı sunar
5. WHEN mobil cihazda tooltip gösterilirse THEN the Frontend SHALL touch-friendly tooltip kullanır

### Requirement 11

**User Story:** Bir kullanıcı olarak, reasoning bilgilerini filtreleyebilmek istiyorum, böylece sadece ilgilendiğim bilgileri görebilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL filtre seçenekleri gösterir
2. WHEN kullanıcı "Sadece Tool Selection" seçerse THEN the Frontend SHALL sadece tool selection bölümünü gösterir
3. WHEN kullanıcı "Sadece Category Matching" seçerse THEN the Frontend SHALL sadece category matching bölümünü gösterir
4. WHEN kullanıcı "Sadece Attention Weights" seçerse THEN the Frontend SHALL sadece attention weights bölümünü gösterir
5. WHEN kullanıcı "Tümünü Göster" seçerse THEN the Frontend SHALL tüm reasoning bileşenlerini gösterir

### Requirement 12

**User Story:** Bir kullanıcı olarak, reasoning bilgilerini karşılaştırabilmek istiyorum, böylece farklı hediyeler arasındaki farkları anlayabilirim.

#### Acceptance Criteria

1. WHEN kullanıcı birden fazla hediye seçerse THEN the Frontend SHALL "Karşılaştır" butonu gösterir
2. WHEN karşılaştırma modu aktifse THEN the Frontend SHALL seçili hediyelerin reasoning'lerini yan yana gösterir
3. WHEN category skorları karşılaştırılırsa THEN the Frontend SHALL skorları aynı chart'ta farklı renklerle gösterir
4. WHEN attention weights karşılaştırılırsa THEN the Frontend SHALL weights'leri overlay chart ile gösterir
5. WHEN karşılaştırma kapatılırsa THEN the Frontend SHALL normal görünüme geri döner

### Requirement 13

**User Story:** Bir test engineer olarak, reasoning görselleştirmelerinin doğru çalıştığını test edebilmek istiyorum, böylece kalite güvencesi sağlayabilirim.

#### Acceptance Criteria

1. WHEN reasoning bileşenleri test edilir THEN the Frontend SHALL her bileşen için unit test içerir
2. WHEN ToolSelectionCard test edilir THEN the Frontend SHALL farklı tool selection data'ları ile render testi yapar
3. WHEN CategoryMatchingChart test edilir THEN the Frontend SHALL farklı skor aralıkları ile render testi yapar
4. WHEN AttentionWeightsChart test edilir THEN the Frontend SHALL normalize edilmiş ve edilmemiş data ile test yapar
5. WHEN ThinkingStepsTimeline test edilir THEN the Frontend SHALL farklı step sayıları ile render testi yapar
6. WHEN snapshot testing uygulanır THEN the Frontend SHALL her bileşen için snapshot test içerir

### Requirement 14

**User Story:** Bir kullanıcı olarak, reasoning bilgilerini export edebilmek istiyorum, böylece daha sonra inceleyebilir veya paylaşabilirim.

#### Acceptance Criteria

1. WHEN detaylı reasoning panel açılır THEN the Frontend SHALL "Export" butonu gösterir
2. WHEN kullanıcı JSON export seçerse THEN the Frontend SHALL reasoning data'yı JSON formatında indirir
3. WHEN kullanıcı PDF export seçerse THEN the Frontend SHALL reasoning görselleştirmelerini PDF olarak indirir
4. WHEN kullanıcı paylaş seçerse THEN the Frontend SHALL reasoning link'ini clipboard'a kopyalar
5. WHEN export başarılı olursa THEN the Frontend SHALL başarı mesajı gösterir

### Requirement 15

**User Story:** Bir kullanıcı olarak, reasoning görselleştirmelerinin erişilebilir olmasını istiyorum, böylece engelli kullanıcılar da sistemi kullanabilir.

#### Acceptance Criteria

1. WHEN reasoning bileşenleri oluşturulur THEN the Frontend SHALL ARIA labels ve roles kullanır
2. WHEN chart'lar gösterilir THEN the Frontend SHALL alt text ve açıklamalar sağlar
3. WHEN klavye navigasyonu kullanılırsa THEN the Frontend SHALL tüm interaktif elementlere erişim sağlar
4. WHEN screen reader kullanılırsa THEN the Frontend SHALL reasoning bilgilerini anlamlı şekilde okur
5. WHEN renk körlüğü varsa THEN the Frontend SHALL renk dışında da görsel ipuçları (pattern, icon) kullanır
