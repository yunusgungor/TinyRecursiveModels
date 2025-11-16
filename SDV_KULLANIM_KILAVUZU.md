# SDV ile Sentetik Veri Üretimi Kılavuzu

## 🎯 Genel Bakış

Bu proje, [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) kullanarak hediye önerisi modeli için sentetik eğitim verisi üretir.

## 📦 Kurulum

### 1. SDV Kütüphanesini Yükleyin

```bash
pip install -r requirements.txt
```

veya sadece SDV için:

```bash
pip install sdv pandas
```

### 2. Temel Veriyi Hazırlayın

Önce mevcut gerçek veriyi oluşturun:

```bash
python create_gift_data.py
```

Bu komut şunları oluşturur:
- `data/realistic_gift_catalog.json` - Gerçek hediye kataloğu
- `data/realistic_user_scenarios.json` - Kullanıcı senaryoları

## 🚀 Kullanım

### Temel Kullanım

```bash
python sdv_data_generator.py
```

Bu komut:
1. Mevcut veriyi yükler
2. SDV synthesizer'ı eğitir
3. 200 sentetik hediye örneği üretir
4. Sonuçları `data/synthetic_gift_catalog.json` dosyasına kaydeder

### Python'dan Kullanım

```python
from sdv_data_generator import GiftDataSynthesizer

# Generator'ı başlat
generator = GiftDataSynthesizer()

# Temel veriyi yükle
base_df = generator.load_base_data()

# Synthesizer'ı eğit
generator.train_synthesizer(base_df, method="gaussian")

# Sentetik veri üret
synthetic_df = generator.generate_synthetic_data(num_samples=500)

# Kaydet
generator.save_synthetic_catalog(synthetic_df, "data/my_synthetic_data.json")
```

## 🔧 Yapılandırma

`config/sdv_config.yaml` dosyasını düzenleyerek ayarları özelleştirebilirsiniz:

```yaml
generation:
  num_synthetic_gifts: 500      # Üretilecek hediye sayısı
  num_synthetic_users: 200      # Üretilecek kullanıcı sayısı
  
synthesizer:
  method: "gaussian"             # veya "ctgan", "tvae"
```

## 📊 SDV Yöntemleri

### 1. Gaussian Copula (Varsayılan)
- **Hızlı** ve hafif
- Basit veri dağılımları için ideal
- Önerilen başlangıç yöntemi

```python
generator.train_synthesizer(base_df, method="gaussian")
```

### 2. CTGAN
- **Daha karmaşık** desenler için
- Daha uzun eğitim süresi
- Daha gerçekçi sonuçlar

```python
generator.train_synthesizer(base_df, method="ctgan")
```

### 3. TVAE
- Tabular veri için optimize edilmiş
- Orta seviye karmaşıklık

## 🎓 Model Eğitimi ile Entegrasyon

### 1. Sentetik Veri Üretin

```bash
python sdv_data_generator.py
```

### 2. Modeli Sentetik Veri ile Eğitin

```bash
python train_integrated_enhanced_model.py \
  --config config/tool_enhanced_gift_recommendation.yaml \
  --data-path data/synthetic_gift_catalog.json
```

### 3. Gerçek ve Sentetik Veriyi Birleştirin

```python
import json

# Gerçek veriyi yükle
with open("data/realistic_gift_catalog.json") as f:
    real_data = json.load(f)

# Sentetik veriyi yükle
with open("data/synthetic_gift_catalog.json") as f:
    synthetic_data = json.load(f)

# Birleştir
combined_gifts = real_data['gifts'] + synthetic_data['gifts']

# Kaydet
with open("data/combined_gift_catalog.json", "w") as f:
    json.dump({"gifts": combined_gifts}, f, indent=2)
```

## 📈 Veri Kalitesi Değerlendirmesi

SDV, üretilen verinin kalitesini değerlendirmek için araçlar sunar:

```python
from sdv.evaluation.single_table import evaluate_quality

# Kalite raporu oluştur
quality_report = evaluate_quality(
    real_data=base_df,
    synthetic_data=synthetic_df,
    metadata=generator.metadata
)

print(quality_report)
```

## 🔍 İleri Seviye Özellikler

### Kısıtlamalar (Constraints)

Belirli kuralları zorunlu kılın:

```python
from sdv.constraints import Inequality

# Fiyat kısıtlaması ekle
constraints = [
    Inequality(
        low_column_name='price',
        high_column_name='price',
        low_value=10.0,
        high_value=500.0
    )
]

synthesizer = GaussianCopulaSynthesizer(
    metadata,
    constraints=constraints
)
```

### Koşullu Örnekleme

Belirli kategoriler için veri üretin:

```python
# Sadece "technology" kategorisi için üret
conditions = pd.DataFrame({
    'category': ['technology'] * 50
})

synthetic_tech = synthesizer.sample_from_conditions(conditions)
```

## 🎯 En İyi Pratikler

1. **Küçük Başlayın**: İlk denemede az sayıda örnek üretin
2. **Kaliteyi Kontrol Edin**: Üretilen veriyi görselleştirin ve inceleyin
3. **Yöntemleri Karşılaştırın**: Farklı synthesizer'ları deneyin
4. **Gerçek Veri ile Karıştırın**: %70 gerçek, %30 sentetik veri kullanın
5. **Düzenli Güncelleyin**: Yeni gerçek veri geldikçe synthesizer'ı yeniden eğitin

## 📚 Kaynaklar

- [SDV Resmi Dokümantasyonu](https://docs.sdv.dev/)
- [SDV GitHub Deposu](https://github.com/sdv-dev/SDV)
- [SDV Örnekleri](https://github.com/sdv-dev/SDV/tree/main/examples)

## 🐛 Sorun Giderme

### SDV Yüklenemiyor

```bash
# Python sürümünüzü kontrol edin (3.8+ gerekli)
python --version

# pip'i güncelleyin
pip install --upgrade pip

# Tekrar deneyin
pip install sdv
```

### Bellek Hatası

Büyük veri setleri için batch boyutunu küçültün:

```python
synthesizer = CTGANSynthesizer(
    metadata,
    batch_size=100  # Varsayılan 500'den küçült
)
```

### Düşük Kaliteli Sentetik Veri

1. Daha fazla gerçek veri toplayın
2. Farklı bir synthesizer deneyin (CTGAN)
3. Eğitim epoch sayısını artırın
4. Metadata yapılandırmasını gözden geçirin

## 💡 İpuçları

- **Hız için**: Gaussian Copula kullanın
- **Kalite için**: CTGAN kullanın (daha uzun sürer)
- **Denge için**: TVAE kullanın
- **Çok kategorili veri için**: CTGAN en iyi sonucu verir
- **Sürekli değişkenler için**: Gaussian Copula yeterlidir

## 🎉 Sonraki Adımlar

1. ✅ SDV'yi kurun ve test edin
2. ✅ Sentetik veri üretin
3. ✅ Kaliteyi değerlendirin
4. ✅ Model eğitiminde kullanın
5. ✅ Sonuçları karşılaştırın
6. ✅ Üretim ortamına deploy edin

Başarılar! 🚀
