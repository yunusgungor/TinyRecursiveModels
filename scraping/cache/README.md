# Cache Directory

Bu klasör Gemini API yanıtlarını önbelleğe alır ve API kullanımını azaltır.

## Cache Dosyaları

- `enhancement_cache.json` - Ürün enhancement'ları
- `scenario_cache.json` - Kullanıcı senaryoları

## Özellikler

- ✅ Otomatik cache yönetimi
- ✅ 30 günlük TTL (Time-to-Live)
- ✅ Disk'e otomatik kaydetme
- ✅ Benzer ürünleri tespit etme

## Cache Temizleme

Cache'i temizlemek için bu dosyaları silebilirsiniz:

```bash
rm scraping/cache/*.json
```
