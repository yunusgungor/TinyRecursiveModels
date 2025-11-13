# ⚡ Hızlı Selector Bulma Rehberi (5 Dakika)

Playwright kuruldu! ✅ Şimdi selector'ları bulalım.

## 🎯 En Basit Yöntem: Manuel Browser İnceleme

### 1️⃣ Web Sitesini Aç

Normal browser'ınızda (Chrome/Edge/Firefox) şu siteyi açın:
```
https://www.ciceksepeti.com/hediye
```

### 2️⃣ Developer Tools'u Aç

**Mac**: `Cmd + Option + I`  
**Windows/Linux**: `F12`

### 3️⃣ Element Seçici'yi Aktifleştir

Developer Tools'da sol üstteki **🔍 ok işaretine** tıklayın.

Veya klavye: `Cmd + Shift + C` (Mac) / `Ctrl + Shift + C` (Windows)

### 4️⃣ Ürün Kartına Tıklayın

Sayfadaki herhangi bir ürün kartına tıklayın. Developer Tools'da HTML vurgulanacak.

### 5️⃣ Selector'ı Kopyalayın

Vurgulanan HTML'de **sağ tık** → **Copy** → **Copy selector**

### 6️⃣ Console'da Test Edin

Developer Tools'da **Console** sekmesine geçin ve yapıştırın:

```javascript
// Kopyaladığınız selector'ı test edin
document.querySelectorAll('BURAYA_YAPIŞTIR').length

// Örnek:
document.querySelectorAll('.product-card').length
// → 24 (24 ürün bulundu demektir)
```

### 7️⃣ Scraper'a Ekleyin

`scraping/scrapers/ciceksepeti_scraper.py` dosyasını açın ve güncelleyin:

```python
SELECTORS = {
    'product_list': '.product-card',  # Buraya yapıştırın
    'product_link': 'a.product-link',
    'product_name': 'h3.product-name',
    'product_price': 'span.product-price',
    'product_description': 'div.product-description',
    'product_image': 'img.product-image',
    'product_rating': 'span.rating',
    'next_page': 'a.next-page',
}
```

---

## 📋 Bulmanız Gereken Selector'lar

### Liste Sayfası (https://www.ciceksepeti.com/hediye)

| Element | Ne Yapmalı | Örnek Selector |
|---------|-----------|----------------|
| **Ürün Kartları** | Bir ürün kartına tıkla → Copy selector | `.product-card` |
| **Ürün Linki** | Ürün kartı içindeki linke tıkla → Copy selector | `a.product-link` |
| **Sonraki Sayfa** | "Sonraki" butonuna tıkla → Copy selector | `a.next-page` |

### Detay Sayfası (Bir ürüne tıklayın)

| Element | Ne Yapmalı | Örnek Selector |
|---------|-----------|----------------|
| **Ürün İsmi** | Başlığa tıkla → Copy selector | `h1.product-title` |
| **Fiyat** | Fiyata tıkla → Copy selector | `span.price` |
| **Açıklama** | Açıklama metnine tıkla → Copy selector | `div.description` |
| **Resim** | Ana resme tıkla → Copy selector | `img.product-img` |
| **Puan** | Yıldızlara tıkla → Copy selector | `span.rating` |

---

## 🧪 Console'da Test Örnekleri

Developer Tools → Console sekmesinde:

```javascript
// Kaç ürün var?
document.querySelectorAll('.product-card').length

// İlk ürünün ismi
document.querySelector('.product-name').textContent

// Tüm fiyatları göster
document.querySelectorAll('.product-price').forEach(el => {
  console.log(el.textContent)
})

// İlk 5 ürünün ismini göster
document.querySelectorAll('.product-name').forEach((el, i) => {
  if (i < 5) console.log(`${i+1}. ${el.textContent}`)
})
```

---

## ✅ Gerçek Örnek: Çiçek Sepeti

### Adım 1: Siteye Git
```
https://www.ciceksepeti.com/hediye
```

### Adım 2: Bir Ürün Kartını İncele

1. `F12` ile Developer Tools'u aç
2. `Cmd/Ctrl + Shift + C` ile element seçiciyi aktifleştir
3. Bir ürün kartına tıkla
4. HTML'de şöyle bir şey göreceksiniz:

```html
<div class="product-card">
  <a href="/product/123">
    <img src="..." />
    <h3 class="product-name">Hediye Paketi</h3>
    <span class="product-price">150,00 TL</span>
  </a>
</div>
```

### Adım 3: Selector'ları Belirle

```python
SELECTORS = {
    'product_list': '.product-card',
    'product_link': '.product-card a',
    'product_name': '.product-name',
    'product_price': '.product-price',
}
```

### Adım 4: Test Et

```bash
python scripts/run_scraping_pipeline.py --test --website ciceksepeti
```

---

## 💡 İpuçları

### ✅ İyi Selector'lar
- `.product-card` - Açık class name
- `[data-product-id]` - Data attribute
- `.products .product-item` - Nested selector

### ❌ Kaçınılacak Selector'lar
- `.css-1a2b3c4` - Dinamik class
- `div:nth-child(5)` - Pozisyona bağlı
- `#product-123` - Spesifik ID

### 🔍 Selector Bulamıyorsanız

1. **Daha genel bir selector deneyin**:
   ```javascript
   // Çok spesifik
   document.querySelector('div.container > div.row > div.col > div.product')
   
   // Daha iyi
   document.querySelector('.product')
   ```

2. **Alternatif selector'lar deneyin**:
   ```javascript
   // Birden fazla seçenek
   document.querySelector('.product-card') ||
   document.querySelector('[data-product]') ||
   document.querySelector('article.product')
   ```

3. **Parent'tan başlayın**:
   ```javascript
   // Önce parent'ı bulun
   const container = document.querySelector('.products-container')
   // Sonra içindeki elementleri
   const products = container.querySelectorAll('.product')
   ```

---

## 🚀 Hızlı Test

Selector'larınızı test etmek için:

```bash
# Test modunda çalıştır (sadece 10 ürün)
python scripts/run_scraping_pipeline.py --test --website ciceksepeti

# Logları kontrol et
tail -f logs/scraping.log
```

---

## 📞 Yardım

Eğer selector bulamıyorsanız:

1. **Console'da test edin**: Selector'ın çalışıp çalışmadığını kontrol edin
2. **Screenshot alın**: Hangi elementi bulmaya çalıştığınızı gösterin
3. **HTML'i inceleyin**: Elementin gerçek yapısına bakın

---

## ✨ Başarı!

Selector'ları buldunuz mu? Harika! Şimdi:

1. ✅ Selector'ları scraper dosyalarına ekleyin
2. ✅ Test modunda çalıştırın
3. ✅ Sonuçları kontrol edin
4. ✅ Gerçek scraping'e başlayın!

İyi scraping'ler! 🎉
