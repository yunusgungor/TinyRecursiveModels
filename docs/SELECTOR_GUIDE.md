# 🎯 Web Selector'larını Güncelleme Rehberi

Bu rehber, gerçek web sitelerinden doğru CSS selector'larını nasıl bulacağınızı ve scraper'lara nasıl ekleyeceğinizi gösterir.

## 📋 İçindekiler
1. [Browser Developer Tools Kullanımı](#browser-developer-tools)
2. [Selector Bulma Adımları](#selector-bulma)
3. [Selector'ları Test Etme](#test-etme)
4. [Scraper'lara Ekleme](#scraperlar-ekleme)
5. [Örnekler](#örnekler)

---

## 🔧 Browser Developer Tools Kullanımı

### Chrome/Edge'de Developer Tools Açma:
- **Windows/Linux**: `F12` veya `Ctrl + Shift + I`
- **Mac**: `Cmd + Option + I`
- **Sağ tık**: "Inspect" veya "Öğeyi İncele"

### Önemli Sekmeler:
- **Elements**: HTML yapısını görüntüle
- **Console**: Selector'ları test et
- **Network**: Sayfa yüklemelerini izle

---

## 🎯 Selector Bulma Adımları

### Adım 1: Web Sitesine Git

Örnek: `https://www.ciceksepeti.com/hediye`

### Adım 2: Developer Tools'u Aç

`F12` veya `Cmd + Option + I`

### Adım 3: Element Seçici Aracını Kullan

1. Developer Tools'da sol üstteki **"Select Element"** ikonuna tıkla (ok işareti)
2. Veya `Ctrl + Shift + C` (Mac: `Cmd + Shift + C`)

### Adım 4: İstediğin Elementi Seç

Sayfada istediğin elemente (ürün kartı, fiyat, isim vb.) tıkla.

### Adım 5: HTML Yapısını İncele

Elements sekmesinde seçili element vurgulanır. Şimdi selector'ı bul:

#### Yöntem 1: Copy Selector (Hızlı)
1. Element üzerinde sağ tık
2. **Copy** → **Copy selector**
3. Selector'ı kopyala

#### Yöntem 2: Manuel İnceleme (Önerilen)
HTML yapısına bakarak daha stabil selector'lar bul:

```html
<!-- Örnek HTML -->
<div class="product-card">
  <a href="/product/123" class="product-link">
    <img src="image.jpg" class="product-image" />
    <h3 class="product-name">Ürün Adı</h3>
    <span class="product-price">150,00 TL</span>
    <div class="product-rating">4.5</div>
  </a>
</div>
```

**İyi Selector'lar**:
- `.product-card` - Class name
- `[data-product-id]` - Data attribute
- `.product-list .product-card` - Nested selector

**Kötü Selector'lar**:
- `div > div > div:nth-child(3)` - Çok spesifik, kırılgan
- `#root > div > div > div` - Dinamik ID'ler

---

## 🧪 Selector'ları Test Etme

### Console'da Test Et

Developer Tools'da **Console** sekmesine git ve şunu yaz:

```javascript
// Tek element
document.querySelector('.product-card')

// Tüm elementler
document.querySelectorAll('.product-card')

// Kaç tane var?
document.querySelectorAll('.product-card').length

// İlk 3 ürünün ismini göster
document.querySelectorAll('.product-name').forEach((el, i) => {
  if (i < 3) console.log(el.textContent)
})
```

### Playwright ile Test Et

Küçük bir test scripti oluştur:

```python
# test_selectors.py
import asyncio
from playwright.async_api import async_playwright

async def test_selectors():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Siteye git
        await page.goto('https://www.ciceksepeti.com/hediye')
        await page.wait_for_load_state('networkidle')
        
        # Selector'ları test et
        products = await page.query_selector_all('.product-card')
        print(f"Bulunan ürün sayısı: {len(products)}")
        
        # İlk ürünün ismini al
        if products:
            name = await products[0].query_selector('.product-name')
            if name:
                text = await name.inner_text()
                print(f"İlk ürün: {text}")
        
        await browser.close()

asyncio.run(test_selectors())
```

Çalıştır:
```bash
python test_selectors.py
```

---

## 📝 Scraper'lara Ekleme

### Örnek: Çiçek Sepeti Selector'larını Güncelleme

#### 1. Dosyayı Aç
```bash
# Editörde aç
code scraping/scrapers/ciceksepeti_scraper.py
```

#### 2. SELECTORS Dictionary'sini Bul

```python
SELECTORS = {
    'product_list': '.product-item',  # Placeholder
    'product_link': 'a.product-link',  # Placeholder
    'product_name': 'h1.product-name',  # Placeholder
    'product_price': '.product-price',  # Placeholder
    'product_description': '.product-description',  # Placeholder
    'product_image': 'img.product-image',  # Placeholder
    'product_rating': '.product-rating',  # Placeholder
    'next_page': '.pagination-next',  # Placeholder
}
```

#### 3. Gerçek Selector'larla Değiştir

Web sitesinden bulduğun selector'ları ekle:

```python
SELECTORS = {
    # Ürün listesi sayfası
    'product_list': 'div[data-testid="product-card"]',  # Gerçek selector
    'product_link': 'a.product-card-link',  # Gerçek selector
    
    # Ürün detay sayfası
    'product_name': 'h1.product-detail-title',  # Gerçek selector
    'product_price': 'span.product-price-value',  # Gerçek selector
    'product_description': 'div.product-description-text',  # Gerçek selector
    'product_image': 'img.product-main-image',  # Gerçek selector
    'product_rating': 'span.rating-score',  # Gerçek selector
    
    # Pagination
    'next_page': 'a.pagination-next-button',  # Gerçek selector
}
```

---

## 💡 Örnekler

### Örnek 1: Çiçek Sepeti

#### Adımlar:
1. `https://www.ciceksepeti.com/hediye` sayfasına git
2. `F12` ile Developer Tools'u aç
3. Bir ürün kartına sağ tık → Inspect

#### Bulunacak Selector'lar:

```python
# Ürün listesi sayfasında
SELECTORS = {
    # Liste sayfası
    'product_list': 'div.product-card',  # Tüm ürün kartları
    'product_link': 'a.product-link',    # Ürün linki
    
    # Detay sayfası (bir ürüne tıklayıp inspect et)
    'product_name': 'h1.product-title',
    'product_price': 'span.price-value',
    'product_description': 'div.description-content',
    'product_image': 'img.main-product-image',
    'product_rating': 'div.rating-stars span',
    
    # Pagination
    'next_page': 'a.next-page-button',
}
```

### Örnek 2: Hepsiburada

```python
SELECTORS = {
    'product_list': 'li.productListContent-item',
    'product_link': 'a.product-link',
    'product_name': 'h1.product-name',
    'product_price': 'span.price-value',
    'product_description': 'div.product-detail-description',
    'product_image': 'img.product-image',
    'product_rating': 'span.rating-score',
    'next_page': 'a.pagination-next',
}
```

### Örnek 3: Trendyol

```python
SELECTORS = {
    'product_list': 'div.p-card-wrppr',
    'product_link': 'a.prdct-desc-cntnr-ttl',
    'product_name': 'h1.pr-new-br',
    'product_price': 'span.prc-dsc',
    'product_description': 'div.detail-desc-wrapper',
    'product_image': 'img.detail-img',
    'product_rating': 'span.ratings-score',
    'next_page': 'a.pagination-next',
}
```

---

## 🔍 Selector Bulma İpuçları

### 1. Stabil Selector'lar Tercih Et

✅ **İyi**:
- Class names: `.product-card`
- Data attributes: `[data-product-id]`
- Semantic tags: `article.product`

❌ **Kötü**:
- Dinamik class'lar: `.css-1a2b3c4`
- Nth-child: `div:nth-child(5)`
- Çok uzun path'ler

### 2. Birden Fazla Selector Dene

Eğer bir selector çalışmazsa, alternatif dene:

```python
# Önce bu dene
name = await page.query_selector('.product-name')

# Çalışmazsa bu
if not name:
    name = await page.query_selector('h1.title')

# Hala yoksa bu
if not name:
    name = await page.query_selector('[data-testid="product-title"]')
```

### 3. Wait for Selector Kullan

Dinamik içerik için bekle:

```python
await page.wait_for_selector('.product-list', timeout=10000)
```

### 4. Multiple Selectors

Birden fazla olası selector tanımla:

```python
SELECTORS = {
    'product_name': [
        'h1.product-title',
        'h1.product-name',
        '[data-testid="product-title"]'
    ]
}
```

---

## 🚀 Hızlı Başlangıç

### 1. Test Script'i Oluştur

```python
# scripts/find_selectors.py
import asyncio
from playwright.async_api import async_playwright

async def find_selectors(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print(f"🌐 Navigating to: {url}")
        await page.goto(url)
        await page.wait_for_load_state('networkidle')
        
        print("\n📋 Page loaded. Now:")
        print("1. Right-click on elements you want to scrape")
        print("2. Select 'Inspect'")
        print("3. Copy the selector")
        print("4. Press Enter when done")
        
        input("\nPress Enter to close browser...")
        await browser.close()

# Test et
asyncio.run(find_selectors('https://www.ciceksepeti.com/hediye'))
```

### 2. Çalıştır ve Selector'ları Bul

```bash
python scripts/find_selectors.py
```

### 3. Bulduğun Selector'ları Ekle

Scraper dosyasındaki `SELECTORS` dictionary'sine ekle.

### 4. Test Et

```bash
python scripts/run_scraping_pipeline.py --test --website ciceksepeti
```

---

## 🎓 Pratik Yapma

### Egzersiz 1: Basit Selector Bulma

1. `https://www.ciceksepeti.com` sayfasına git
2. Bir ürün kartı bul
3. Ürün isminin selector'ını bul
4. Console'da test et

### Egzersiz 2: Tüm Selector'ları Bul

Bir site için tüm gerekli selector'ları bul:
- [ ] Ürün listesi
- [ ] Ürün linki
- [ ] Ürün ismi
- [ ] Fiyat
- [ ] Açıklama
- [ ] Resim
- [ ] Rating
- [ ] Next page button

### Egzersiz 3: Scraper'ı Test Et

1. Selector'ları ekle
2. Test modunda çalıştır
3. Logları kontrol et
4. Gerekirse düzelt

---

## 🆘 Sorun Giderme

### "Selector not found" Hatası

**Çözüm 1**: Sayfanın yüklenmesini bekle
```python
await page.wait_for_selector('.product-list', timeout=10000)
```

**Çözüm 2**: Farklı selector dene
```python
# Alternatif selector'lar
selectors = ['.product-card', '[data-product]', 'article.product']
for selector in selectors:
    element = await page.query_selector(selector)
    if element:
        break
```

**Çözüm 3**: Screenshot al ve kontrol et
```python
await page.screenshot(path='debug.png')
```

### Dinamik İçerik

Eğer içerik JavaScript ile yükleniyorsa:

```python
# Network idle bekle
await page.wait_for_load_state('networkidle')

# Veya belirli bir süre bekle
await page.wait_for_timeout(2000)

# Veya belirli bir elementi bekle
await page.wait_for_selector('.product-list')
```

### CAPTCHA veya Bot Detection

Eğer CAPTCHA çıkarsa:
1. Rate limit'i düşür
2. Delay'i artır
3. User agent'ı değiştir
4. Headless mode'u kapat (test için)

---

## 📚 Kaynaklar

- [Playwright Selectors](https://playwright.dev/python/docs/selectors)
- [CSS Selectors Reference](https://www.w3schools.com/cssref/css_selectors.asp)
- [Chrome DevTools Guide](https://developer.chrome.com/docs/devtools/)

---

## ✅ Checklist

Selector'ları güncellerken:

- [ ] Browser Developer Tools'u açtım
- [ ] Her element için selector buldum
- [ ] Console'da test ettim
- [ ] Scraper dosyasına ekledim
- [ ] Test modunda çalıştırdım
- [ ] Logları kontrol ettim
- [ ] Gerekirse düzelttim

Başarılar! 🎉
