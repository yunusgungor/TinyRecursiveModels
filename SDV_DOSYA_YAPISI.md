# 📁 SDV Entegrasyonu - Dosya Yapısı

## 🎯 Oluşturulan Dosyalar

### Ana Scriptler

| Dosya | Boyut | Açıklama | Kullanım |
|-------|-------|----------|----------|
| `sdv_data_generator.py` | 5.8KB | Temel sentetik veri üretici | `python sdv_data_generator.py` |
| `sdv_advanced_generator.py` | 13KB | Gelişmiş üretici + kalite kontrolü | `python sdv_advanced_generator.py` |
| `example_sdv_usage.py` | 8.2KB | 5 farklı kullanım örneği | `python example_sdv_usage.py` |
| `setup_sdv.sh` | 1.5KB | Otomatik kurulum scripti | `./setup_sdv.sh` |

### Dokümantasyon

| Dosya | Boyut | Açıklama |
|-------|-------|----------|
| `SDV_README.md` | 4.5KB | Hızlı başlangıç kılavuzu |
| `SDV_KULLANIM_KILAVUZU.md` | 5.7KB | Detaylı Türkçe kılavuz |
| `SDV_DOSYA_YAPISI.md` | Bu dosya | Dosya yapısı ve özet |

### Yapılandırma

| Dosya | Boyut | Açıklama |
|-------|-------|----------|
| `config/sdv_config.yaml` | 1.6KB | SDV yapılandırma dosyası |

### Güncellenmiş Dosyalar

| Dosya | Değişiklik |
|-------|------------|
| `requirements.txt` | SDV ve pandas eklendi |

## 🗂️ Proje Yapısı

```
TinyRecursiveModels/
├── sdv_data_generator.py          # Temel üretici
├── sdv_advanced_generator.py      # Gelişmiş üretici
├── example_sdv_usage.py           # Örnekler
├── setup_sdv.sh                   # Kurulum scripti
├── SDV_README.md                  # Hızlı başlangıç
├── SDV_KULLANIM_KILAVUZU.md      # Detaylı kılavuz
├── SDV_DOSYA_YAPISI.md           # Bu dosya
├── requirements.txt               # Güncellenmiş bağımlılıklar
├── config/
│   ├── sdv_config.yaml           # SDV yapılandırması
│   └── ...
├── data/                          # Veri klasörü (oluşturulacak)
│   ├── realistic_gift_catalog.json
│   ├── realistic_user_scenarios.json
│   ├── synthetic_gift_catalog.json
│   ├── synthetic_user_scenarios.json
│   └── sdv_quality_report.json
└── ...
```

## 🚀 Kullanım Akışı

### 1. Kurulum
```bash
./setup_sdv.sh
```
**Çıktı:**
- SDV kurulumu
- Temel veri oluşturma
- İlk sentetik veri üretimi

### 2. Temel Kullanım
```bash
python sdv_data_generator.py
```
**Çıktı:**
- `data/synthetic_gift_catalog.json` (200 örnek)

### 3. Gelişmiş Kullanım
```bash
python sdv_advanced_generator.py
```
**Çıktı:**
- `data/synthetic_gift_catalog.json` (300 hediye)
- `data/synthetic_user_scenarios.json` (150 kullanıcı)
- `data/sdv_quality_report.json` (kalite raporu)

### 4. Örnekleri İncele
```bash
python example_sdv_usage.py
```
**5 örnek:**
1. Temel sentetik veri üretimi
2. Kısıtlamalarla veri üretimi
3. Kalite değerlendirmesi
4. Koşullu örnekleme
5. Yöntem karşılaştırması

## 📊 Özellik Karşılaştırması

| Özellik | Temel | Gelişmiş |
|---------|-------|----------|
| Hız | ⚡⚡⚡ | ⚡⚡ |
| Kalite Kontrolü | ❌ | ✅ |
| Çoklu Synthesizer | ❌ | ✅ |
| Kısıtlamalar | ❌ | ✅ |
| Kalite Raporu | ❌ | ✅ |
| Kullanıcı Verisi | ❌ | ✅ |
| Otomatik En İyi Seçim | ❌ | ✅ |

## 🎓 Öğrenme Yolu

### Seviye 1: Başlangıç
1. `SDV_README.md` okuyun
2. `./setup_sdv.sh` çalıştırın
3. `python sdv_data_generator.py` deneyin

### Seviye 2: Orta
1. `example_sdv_usage.py` örneklerini inceleyin
2. `config/sdv_config.yaml` dosyasını özelleştirin
3. Farklı synthesizer yöntemlerini deneyin

### Seviye 3: İleri
1. `SDV_KULLANIM_KILAVUZU.md` detaylı kılavuzu okuyun
2. `sdv_advanced_generator.py` kullanın
3. Kendi kısıtlamalarınızı ekleyin
4. Kalite metriklerini optimize edin

## 🔧 Yapılandırma Seçenekleri

### `config/sdv_config.yaml`

```yaml
# Üretim miktarları
generation:
  num_synthetic_gifts: 500
  num_synthetic_users: 200

# Synthesizer yöntemi
synthesizer:
  method: "gaussian"  # veya "ctgan", "tvae"

# Kalite kısıtlamaları
constraints:
  price_min: 10.0
  price_max: 500.0
  rating_min: 3.0
  rating_max: 5.0
```

## 📈 Performans Beklentileri

| Yöntem | Eğitim Süresi | Üretim Süresi | Kalite |
|--------|---------------|---------------|--------|
| Gaussian Copula | ~5 saniye | ~1 saniye | ⭐⭐ |
| CTGAN | ~2-5 dakika | ~5 saniye | ⭐⭐⭐ |
| TVAE | ~1-3 dakika | ~3 saniye | ⭐⭐⭐ |

*100 örnek için, CPU üzerinde*

## 🎯 Kullanım Senaryoları

### Senaryo 1: Hızlı Prototipleme
```bash
python sdv_data_generator.py
```
- Hızlı sonuç
- Temel kalite
- İlk testler için ideal

### Senaryo 2: Üretim Ortamı
```bash
python sdv_advanced_generator.py
```
- Yüksek kalite
- Kalite kontrolü
- Gerçek kullanım için

### Senaryo 3: Araştırma & Geliştirme
```bash
python example_sdv_usage.py
```
- Farklı yöntemleri test et
- Parametreleri optimize et
- En iyi yaklaşımı bul

## 💾 Veri Çıktıları

### Üretilen Veri Dosyaları

| Dosya | İçerik | Boyut (tahmini) |
|-------|--------|-----------------|
| `realistic_gift_catalog.json` | Gerçek hediye verisi | ~15KB |
| `realistic_user_scenarios.json` | Gerçek kullanıcı verisi | ~8KB |
| `synthetic_gift_catalog.json` | Sentetik hediye verisi | ~30KB |
| `synthetic_user_scenarios.json` | Sentetik kullanıcı verisi | ~15KB |
| `sdv_quality_report.json` | Kalite raporu | ~5KB |

## 🔍 Kalite Metrikleri

Kalite raporu şunları içerir:

1. **Overall Score**: Genel kalite skoru (0-1)
2. **Column Shapes**: Sütun dağılımları benzerliği
3. **Column Pair Trends**: Sütun çiftleri arasındaki ilişkiler
4. **Statistical Similarity**: İstatistiksel benzerlik

## 🎉 Başarı Kriterleri

✅ **İyi Kalite** (>0.80):
- Model eğitimi için kullanılabilir
- Gerçek veri ile karıştırılabilir

⚠️ **Orta Kalite** (0.60-0.80):
- Ek veri olarak kullanılabilir
- Gerçek veri ile birleştirilmeli

❌ **Düşük Kalite** (<0.60):
- Parametreleri ayarlayın
- Daha fazla gerçek veri toplayın
- Farklı synthesizer deneyin

## 📞 Destek

Sorun yaşarsanız:

1. `SDV_KULLANIM_KILAVUZU.md` → Sorun Giderme bölümü
2. [SDV Dokümantasyonu](https://docs.sdv.dev/)
3. [SDV GitHub Issues](https://github.com/sdv-dev/SDV/issues)

---

**Hazır mısınız?** `./setup_sdv.sh` ile başlayın! 🚀
