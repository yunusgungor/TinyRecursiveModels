# 🎨 Görsel Selector Bulma Rehberi

## 🚀 Hızlı Başlangıç (5 Dakika)

### Yöntem 1: Otomatik Selector Finder (Önerilen)

```bash
# Helper script'i çalıştır
python scripts/find_selectors.py https://www.ciceksepeti.com/hediye

# Veya interaktif mod
python scripts/find_selectors.py --interactive
```

Script otomatik olarak:
- ✅ Sayfayı açar
- ✅ Yaygın selector'ları test eder
- ✅ Bulunan selector'ları gösterir
- ✅ Browser'ı açık tutar (manuel inceleme için)

### Yöntem 2: Manuel (Adım Adım)

## 📸 Adım Adım Görsel Rehber

### 1️⃣ Web Sitesine Git

```
🌐 https://www.ciceksepeti.com/hediye
```

### 2️⃣ Developer Tools'u Aç

**Mac**: `Cmd + Option + I`
**Windows/Linux**: `F12` veya `Ctrl + Shift + I`

```
┌─────────────────────────────────────────┐
│  🌐 Web Sitesi                          │
│  ┌───────────────────────────────────┐  │
│  │                                   │  │
│  │  [Ürün 1]  [Ürün 2]  [Ürün 3]   │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 🔧 Developer Tools              │   │
│  │ Elements | Console | Network    │   │
│  │                                 │   │
│  │ <div class="product-card">     │   │
│  │   <h3>Ürün Adı</h3>           │   │
│  │   <span>150 TL</span>         │   │
│  │ </div>                         │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 3️⃣ Element Seçici'yi Aktifleştir

Developer Tools'da sol üstteki **ok işaretine** tıkla:

```
┌──────────────────────────────┐
│ 🔧 Developer Tools           │
│ [🔍] Elements Console ...    │  ← Bu oka tıkla
└──────────────────────────────┘
```

Veya klavye kısayolu:
- **Mac**: `Cmd + Shift + C`
- **Windows/Linux**: `Ctrl + Shift + C`

### 4️⃣ Elementi Seç

Sayfada istediğin elemente tıkla (örn: ürün kartı, fiyat, isim)

```
Sayfada:
┌─────────────────────┐
│  🎁 Hediye Paketi   │  ← Buraya tıkla
│  ⭐⭐⭐⭐⭐ 4.5      │
│  💰 150,00 TL       │
└─────────────────────┘

Developer Tools'da:
<div class="product-card">  ← Bu vurgulanır
  <h3 class="product-name">Hediye Paketi</h3>
  <span class="product-price">150,00 TL</span>
</div>
```

### 5️⃣ Selector'ı Kopyala

Vurgulanan HTML üzerinde **sağ tık**:

```
┌──────────────────────────────┐
│ Copy                      ▶  │
│ ├─ Copy outerHTML            │
│ ├─ Copy selector         ✓   │  ← Bunu seç
│ ├─ Copy JS path              │
│ └─ Copy XPath                │
└──────────────────────────────┘
```

Kopyalanan: `.product-card`

### 6️⃣ Console'da Test Et

Developer Tools'da **Console** sekmesine geç:

```javascript
// Tek element
document.querySelector('.product-card')

// Tüm elementler
document.querySelectorAll('.product-card')

// Kaç tane var?
document.querySelectorAll('.product-card').length
// → 24

// İsimleri göster
document.querySelectorAll('.product-name').forEach(el => {
  console.log(el.textContent)
})
```

### 7️⃣ Scraper'a Ekle

`scraping/scrapers/ciceksepeti_scraper.py` dosyasını aç:

```python
SELECTORS = {
    'product_list': '.product-card',        # ← Buraya ekle
    'product_link': 'a.product-link',
    'product_name': 'h3.product-name',      # ← Buraya ekle
    'product_price': 'span.product-price',  # ← Buraya ekle
    # ... diğerleri
}
```

---

## 🎯 Hangi Selector'ları Bulmalıyım?

### Liste Sayfası (Kategori Sayfası)

```
https://www.ciceksepeti.com/hediye
```

Bulunacaklar:
- ✅ `product_list` - Tüm ürün kartları
- ✅ `product_link` - Ürün detay sayfasına link
- ✅ `next_page` - Sonraki sayfa butonu

### Detay Sayfası (Ürün Sayfası)

```
https://www.ciceksepeti.com/product/123
```

Bulunacaklar:
- ✅ `product_name` - Ürün ismi
- ✅ `product_price` - Fiyat
- ✅ `product_description` - Açıklama
- ✅ `product_image` - Ana resim
- ✅ `product_rating` - Puan/yıldız

---

## 🔍 Selector Örnekleri

### Örnek 1: Ürün Kartı

**HTML**:
```html
<div class="product-card" data-product-id="123">
  <a href="/product/123">
    <img src="image.jpg" />
    <h3>Ürün Adı</h3>
    <span class="price">150 TL</span>
  </a>
</div>
```

**Selector'lar**:
```python
'product_list': '.product-card'
'product_link': '.product-card a'
'product_name': '.product-card h3'
'product_price': '.product-card .price'
```

### Örnek 2: Nested Selector

**HTML**:
```html
<div class="products-container">
  <div class="product-item">
    <div class="product-info">
      <h3 class="title">Ürün</h3>
    </div>
  </div>
</div>
```

**Selector'lar**:
```python
'product_list': '.products-container .product-item'
'product_name': '.product-item .title'
```

### Örnek 3: Data Attribute

**HTML**:
```html
<div data-testid="product-card">
  <span data-testid="product-price">150 TL</span>
</div>
```

**Selector'lar**:
```python
'product_list': '[data-testid="product-card"]'
'product_price': '[data-testid="product-price"]'
```

---

## 🧪 Test Etme

### Console'da Hızlı Test

```javascript
// 1. Selector'ı test et
document.querySelectorAll('.product-card').length
// → 24 (24 ürün bulundu)

// 2. İlk ürünün ismini al
document.querySelector('.product-name').textContent
// → "Hediye Paketi"

// 3. Tüm fiyatları göster
document.querySelectorAll('.product-price').forEach(el => {
  console.log(el.textContent)
})
// → "150,00 TL"
// → "200,00 TL"
// → ...
```

### Python ile Test

```python
# test_my_selectors.py
import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto('https://www.ciceksepeti.com/hediye')
        await page.wait_for_load_state('networkidle')
        
        # Test selector
        products = await page.query_selector_all('.product-card')
        print(f"✅ Bulunan ürün: {len(products)}")
        
        # İlk ürünün ismini al
        if products:
            name_el = await products[0].query_selector('.product-name')
            if name_el:
                name = await name_el.inner_text()
                print(f"✅ İlk ürün: {name}")
        
        input("Press Enter to close...")
        await browser.close()

asyncio.run(test())
```

---

## 💡 İpuçları

### ✅ İyi Selector'lar

```python
# Class name
'.product-card'

# Data attribute
'[data-product-id]'

# Semantic HTML
'article.product'

# Nested
'.product-list .product-item'
```

### ❌ Kötü Selector'lar

```python
# Dinamik class (değişebilir)
'.css-1a2b3c4'

# Çok spesifik (kırılgan)
'div > div > div:nth-child(3) > span'

# ID (her sayfada farklı olabilir)
'#product-123'
```

### 🎯 Selector Önceliği

1. **Data attributes** (en stabil)
   ```python
   '[data-testid="product"]'
   ```

2. **Semantic class names**
   ```python
   '.product-card'
   ```

3. **Nested selectors**
   ```python
   '.products .product-item'
   ```

4. **Tag + class**
   ```python
   'article.product'
   ```

---

## 🚨 Yaygın Sorunlar

### Sorun 1: "Selector not found"

**Sebep**: Sayfa henüz yüklenmedi

**Çözüm**:
```python
# Selector'ı bekle
await page.wait_for_selector('.product-card', timeout=10000)
```

### Sorun 2: Dinamik İçerik

**Sebep**: JavaScript ile yükleniyor

**Çözüm**:
```python
# Network idle bekle
await page.wait_for_load_state('networkidle')

# Veya belirli süre bekle
await page.wait_for_timeout(2000)
```

### Sorun 3: Çok Fazla Element

**Sebep**: Selector çok genel

**Çözüm**:
```python
# Daha spesifik selector kullan
# Kötü:
'.card'  # Tüm kartlar

# İyi:
'.product-list .product-card'  # Sadece ürün kartları
```

---

## 📋 Checklist

Her site için bu selector'ları bul:

### Liste Sayfası
- [ ] `product_list` - Ürün kartları
- [ ] `product_link` - Ürün linki
- [ ] `next_page` - Sonraki sayfa

### Detay Sayfası
- [ ] `product_name` - İsim
- [ ] `product_price` - Fiyat
- [ ] `product_description` - Açıklama
- [ ] `product_image` - Resim
- [ ] `product_rating` - Puan

### Test
- [ ] Console'da test ettim
- [ ] Python script ile test ettim
- [ ] Scraper'a ekledim
- [ ] Pipeline'ı çalıştırdım

---

## 🎓 Pratik Yapma

### Egzersiz 1: Basit Site

1. `https://www.ciceksepeti.com/hediye` sayfasına git
2. Developer Tools'u aç
3. Bir ürün kartının selector'ını bul
4. Console'da test et

### Egzersiz 2: Helper Script

```bash
# Otomatik selector finder'ı çalıştır
python scripts/find_selectors.py https://www.ciceksepeti.com/hediye
```

### Egzersiz 3: Tam Scraper

1. Tüm selector'ları bul
2. Scraper'a ekle
3. Test modunda çalıştır
4. Sonuçları kontrol et

---

## 🎉 Başarılı!

Artık selector bulma konusunda uzman oldunuz! 

**Sonraki adımlar**:
1. ✅ Playwright'i kur: `playwright install chromium`
2. ✅ Selector'ları bul ve ekle
3. ✅ Test et: `python scripts/run_scraping_pipeline.py --test`
4. ✅ Gerçek scraping yap!

İyi scraping'ler! 🚀
