# User Scenario Generator

Scraping ile elde edilen **gerçek** gift catalog verilerinden dinamik ve gerçekçi kullanıcı senaryoları oluşturur.

## Özellikler

- ✅ **Gerçek veriden dinamik üretim**: Scraped catalog'dan kategoriler, tag'ler, fiyat aralıkları ve özel günler çıkarılır
- ✅ **Gemini AI entegrasyonu**: Akıllı ve çeşitli senaryo üretimi
- ✅ **Fallback rule-based generation**: API olmadan da çalışır, gerçek veriyi kullanır
- ✅ **Otomatik pipeline entegrasyonu**: Scraping pipeline'ın bir parçası olarak çalışır
- ✅ **Çeşitli profiller**: Yaş grupları, bütçeler, ilişkiler ve tercihler

## Kullanım

### 1. Otomatik (Scraping Pipeline ile)

Scraping pipeline çalıştırıldığında otomatik olarak user scenarios üretilir:

```bash
python scripts/scraping.py
```

Pipeline aşamaları:
1. Web scraping
2. Veri validasyonu
3. Gemini ile enhancement
4. Dataset oluşturma
5. **User scenarios üretimi** (yeni!)

### 2. Manuel Test (Standalone)

Sadece scenario generation'ı test etmek için:

```bash
python scraping/scripts/test_scenario_generator.py
```

### 3. Özel Standalone Script

Eski standalone script (artık önerilmiyor):

```bash
python scraping/scripts/generate_user_scenarios.py 100
```

### 4. Gemini API ile (Opsiyonel)

Daha gerçekçi senaryolar için Gemini API kullanabilirsiniz:

```bash
export GEMINI_API_KEY="your-api-key-here"
python scripts/scraping.py
```

## Çıktı Formatı

```json
{
  "scenarios": [
    {
      "id": "scenario_0000",
      "profile": {
        "age": 25,
        "hobbies": ["technology", "fitness"],
        "relationship": "friend",
        "budget": 150.0,
        "occasion": "birthday",
        "preferences": ["trendy", "practical"]
      },
      "expected_categories": ["technology", "fitness"],
      "expected_tools": ["price_comparison", "review_analysis"]
    }
  ],
  "metadata": {
    "total_scenarios": 100,
    "age_range": {"min": 16, "max": 70, "avg": 35.2},
    "budget_range": {"min": 50, "max": 500, "avg": 180.5}
  }
}
```

## Dinamik Veri Kullanımı

Generator, scraped catalog'dan şu verileri otomatik çıkarır:

- **Kategoriler**: Gerçek ürün kategorileri
- **Tag'ler**: Ürünlerdeki emotional_tags (hobi ve tercih olarak kullanılır)
- **Özel Günler**: Ürünlerdeki occasions listesi
- **Fiyat Aralıkları**: Gerçek ürün fiyatlarından hesaplanan tier'lar (low, medium, high, premium)

Bu sayede üretilen senaryolar **gerçek catalog ile tam uyumlu** olur.

## Konfigürasyon

`scraping/config/scraping_config.yaml`:

```yaml
output:
  final_dataset_path: "data/scraped_gift_catalog.json"
  user_scenarios_path: "data/user_scenarios.json"
  num_user_scenarios: 100  # Üretilecek senaryo sayısı
```

## Gereksinimler

- Gift catalog dosyası: `data/scraped_gift_catalog.json` (scraping pipeline tarafından oluşturulur)
- Python 3.8+
- (Opsiyonel) google-generativeai paketi

## Notlar

- ✅ API anahtarı yoksa otomatik olarak rule-based generation kullanılır (yine gerçek veriyi kullanır)
- ✅ Senaryolar gerçek catalog'daki fiyat aralıklarını kullanır
- ✅ Hobi ve tercihler gerçek ürün tag'lerinden gelir
- ✅ Özel günler gerçek ürün occasions'larından gelir
- ✅ Her senaryo farklı ilişki tipleri ve yaş grupları içerir
