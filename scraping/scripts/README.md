# User Scenario Generator

Scraping ile elde edilen gift catalog verilerinden gerçekçi kullanıcı senaryoları oluşturur.

## Özellikler

- ✅ Gift catalog'dan otomatik kategori çıkarımı
- ✅ Gemini AI ile akıllı senaryo üretimi
- ✅ Fallback rule-based generation (API olmadan da çalışır)
- ✅ Çeşitli yaş grupları, bütçeler ve ilişkiler
- ✅ Gerçekçi kullanıcı profilleri

## Kullanım

### 1. Temel Kullanım (100 senaryo)

```bash
python scraping/scripts/generate_user_scenarios.py
```

### 2. Özel Sayıda Senaryo

```bash
python scraping/scripts/generate_user_scenarios.py 200
```

### 3. Gemini API ile (Opsiyonel)

Daha gerçekçi senaryolar için Gemini API kullanabilirsiniz:

```bash
export GEMINI_API_KEY="your-api-key-here"
python scraping/scripts/generate_user_scenarios.py 100
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

## Gereksinimler

- Gift catalog dosyası: `data/scraped_gift_catalog.json`
- Python 3.8+
- (Opsiyonel) google-generativeai paketi

## Notlar

- API anahtarı yoksa otomatik olarak rule-based generation kullanılır
- Senaryolar çeşitli yaş grupları (16-70) ve bütçeler (50-500 TL) içerir
- Her senaryo farklı ilişki tipleri ve özel günler içerir
