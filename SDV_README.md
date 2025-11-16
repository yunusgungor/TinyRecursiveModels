# 🎁 SDV Entegrasyonu - Hızlı Başlangıç

## 🚀 Hızlı Kurulum (3 Adım)

### 1. SDV'yi Kurun
```bash
pip install sdv pandas
```

### 2. Otomatik Kurulum Scripti
```bash
chmod +x setup_sdv.sh
./setup_sdv.sh
```

### 3. Sentetik Veri Üretin
```bash
python sdv_data_generator.py
```

## 📁 Oluşturulan Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `sdv_data_generator.py` | Temel sentetik veri üretici |
| `sdv_advanced_generator.py` | Gelişmiş üretici (kalite kontrolü ile) |
| `example_sdv_usage.py` | Kullanım örnekleri |
| `config/sdv_config.yaml` | Yapılandırma dosyası |
| `SDV_KULLANIM_KILAVUZU.md` | Detaylı Türkçe kılavuz |

## 🎯 Kullanım Senaryoları

### Senaryo 1: Hızlı Veri Üretimi
```bash
python sdv_data_generator.py
```
**Çıktı:** `data/synthetic_gift_catalog.json` (200 sentetik hediye)

### Senaryo 2: Yüksek Kaliteli Veri
```bash
python sdv_advanced_generator.py
```
**Çıktı:** 
- `data/synthetic_gift_catalog.json` (300 hediye)
- `data/synthetic_user_scenarios.json` (150 kullanıcı)
- `data/sdv_quality_report.json` (kalite raporu)

### Senaryo 3: Örnekleri İnceleyin
```bash
python example_sdv_usage.py
```
**5 farklı kullanım örneği gösterir**

## 💡 Hızlı Python Kullanımı

```python
from sdv_data_generator import GiftDataSynthesizer

# Başlat
generator = GiftDataSynthesizer()

# Veri yükle ve eğit
base_df = generator.load_base_data()
generator.train_synthesizer(base_df, method="gaussian")

# Üret
synthetic_df = generator.generate_synthetic_data(num_samples=500)

# Kaydet
generator.save_synthetic_catalog(synthetic_df, "my_data.json")
```

## 🔧 Yapılandırma

`config/sdv_config.yaml` dosyasını düzenleyin:

```yaml
generation:
  num_synthetic_gifts: 500    # Üretilecek hediye sayısı
  num_synthetic_users: 200    # Üretilecek kullanıcı sayısı

synthesizer:
  method: "gaussian"          # veya "ctgan", "tvae"
```

## 📊 SDV Yöntemleri

| Yöntem | Hız | Kalite | Kullanım |
|--------|-----|--------|----------|
| **Gaussian Copula** | ⚡⚡⚡ | ⭐⭐ | Hızlı prototipleme |
| **CTGAN** | ⚡ | ⭐⭐⭐ | Üretim ortamı |
| **TVAE** | ⚡⚡ | ⭐⭐⭐ | Dengeli seçim |

## 🎓 Model Eğitimi ile Entegrasyon

### Adım 1: Sentetik Veri Üret
```bash
python sdv_advanced_generator.py
```

### Adım 2: Gerçek ve Sentetik Veriyi Birleştir
```python
import json

# Gerçek veri
with open("data/realistic_gift_catalog.json") as f:
    real = json.load(f)

# Sentetik veri
with open("data/synthetic_gift_catalog.json") as f:
    synthetic = json.load(f)

# Birleştir
combined = {
    "gifts": real['gifts'] + synthetic['gifts']
}

# Kaydet
with open("data/combined_catalog.json", "w") as f:
    json.dump(combined, f, indent=2)
```

### Adım 3: Model Eğitimi
```bash
python train_integrated_enhanced_model.py \
  --config config/tool_enhanced_gift_recommendation.yaml \
  --data-path data/combined_catalog.json
```

## 📈 Kalite Kontrolü

Üretilen verinin kalitesini kontrol edin:

```python
from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=real_df,
    synthetic_data=synthetic_df,
    metadata=metadata
)

print(f"Kalite Skoru: {quality_report.get_score():.2%}")
```

## 🐛 Sorun Giderme

### SDV Kurulamıyor
```bash
# Python sürümünü kontrol edin (3.8+ gerekli)
python --version

# pip'i güncelleyin
pip install --upgrade pip

# Tekrar deneyin
pip install sdv
```

### Bellek Hatası
```python
# Batch boyutunu küçültün
synthesizer = CTGANSynthesizer(
    metadata,
    batch_size=100  # Varsayılan: 500
)
```

### Düşük Kalite
1. Daha fazla gerçek veri toplayın
2. CTGAN kullanın (daha yavaş ama kaliteli)
3. Epoch sayısını artırın
4. Metadata yapılandırmasını kontrol edin

## 📚 Kaynaklar

- [SDV Dokümantasyonu](https://docs.sdv.dev/)
- [SDV GitHub](https://github.com/sdv-dev/SDV)
- [Detaylı Kılavuz](SDV_KULLANIM_KILAVUZU.md)

## ✅ Kontrol Listesi

- [ ] SDV kuruldu (`pip install sdv`)
- [ ] Temel veri oluşturuldu (`python create_gift_data.py`)
- [ ] Sentetik veri üretildi (`python sdv_data_generator.py`)
- [ ] Kalite kontrol edildi
- [ ] Model eğitiminde kullanıldı
- [ ] Sonuçlar karşılaştırıldı

## 🎉 Sonraki Adımlar

1. ✅ Temel kurulumu tamamlayın
2. ✅ Örnekleri inceleyin (`example_sdv_usage.py`)
3. ✅ Gelişmiş özellikleri deneyin (`sdv_advanced_generator.py`)
4. ✅ Kendi veri setinizi oluşturun
5. ✅ Model performansını karşılaştırın

---

**İpucu:** Hızlı başlangıç için `./setup_sdv.sh` scriptini çalıştırın! 🚀
