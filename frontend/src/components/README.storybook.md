# Storybook Documentation - Reasoning Visualization Components

Bu doküman, Trendyol Gift Recommendation sisteminin reasoning visualization bileşenleri için Storybook dokümantasyonunu içerir.

## Genel Bakış

Reasoning visualization bileşenleri, backend'den gelen model reasoning bilgilerini (düşünme adımları, tool seçim mantığı, kategori eşleştirme, attention weights) kullanıcı dostu bir şekilde görselleştirir.

## Storybook'u Çalıştırma

```bash
cd frontend
npm run storybook
```

Storybook, varsayılan olarak `http://localhost:6006` adresinde açılacaktır.

## Bileşen Kategorileri

### 1. Temel Görselleştirme Bileşenleri

#### GiftRecommendationCard
**Dosya:** `GiftRecommendationCard.stories.tsx`

Hediye önerilerini reasoning açıklamalarıyla birlikte gösteren kart bileşeni.

**Öne Çıkan Stories:**
- `Default`: Standart hediye kartı
- `HighConfidence`: Yüksek güven skorlu kart
- `LowConfidence`: Düşük güven skorlu kart
- `LongReasoning`: Uzun reasoning metni ile
- `Selected`: Seçili durum (karşılaştırma modu için)

**Props:**
- `recommendation`: Hediye önerisi ve reasoning bilgileri
- `toolResults`: Tool sonuçları (rating, trend, stok)
- `onShowDetails`: Detaylı panel açma callback'i
- `isSelected`: Seçili durumu
- `onSelect`: Seçim değiştirme callback'i

**Erişilebilirlik:**
- ARIA labels tüm interaktif elementlerde
- Klavye navigasyonu desteklenir
- Renk körlüğü dostu tasarım (icon'lar + renkler)

---

#### ConfidenceIndicator
**Dosya:** `ConfidenceIndicator.stories.tsx`

Güven skorunu görsel olarak gösteren badge bileşeni.

**Öne Çıkan Stories:**
- `HighConfidence`: >0.8 (yeşil)
- `MediumConfidence`: 0.5-0.8 (sarı)
- `LowConfidence`: <0.5 (kırmızı)
- `MultipleIndicators`: Farklı skorları yan yana gösterir

**Renk Kodlaması:**
- 🟢 Yeşil (>0.8): Yüksek Güven
- 🟡 Sarı (0.5-0.8): Orta Güven
- 🔴 Kırmızı (<0.5): Düşük Güven

**Props:**
- `confidence`: 0-1 arası güven skoru
- `onClick`: Açıklama modal'ı açma callback'i (opsiyonel)

---

#### ConfidenceExplanationModal
**Dosya:** `ConfidenceExplanationModal.stories.tsx`

Güven skorunun detaylı açıklamasını gösteren modal.

**Öne Çıkan Stories:**
- `HighConfidence`: Çoğunlukla pozitif faktörler
- `MediumConfidence`: Karışık faktörler
- `LowConfidence`: Çoğunlukla negatif faktörler
- `OnlyPositiveFactors`: Sadece pozitif faktörler
- `OnlyNegativeFactors`: Sadece negatif faktörler

**Props:**
- `isOpen`: Modal açık/kapalı durumu
- `onClose`: Modal kapatma callback'i
- `explanation`: Güven açıklaması (skor, seviye, faktörler)

---

### 2. Tool ve Kategori Görselleştirmeleri

#### ToolSelectionCard
**Dosya:** `ToolSelectionCard.stories.tsx`

Hangi tool'ların neden seçildiğini gösteren kart.

**Öne Çıkan Stories:**
- `Default`: Karışık seçim durumları
- `AllSelected`: Tüm tool'lar seçili
- `AllUnselected`: Hiçbir tool seçili değil
- `LowConfidence`: Düşük güvenli tool'lar

**Görsel İpuçları:**
- ✅ Yeşil + Checkmark: Seçili tool
- ⚪ Gri: Seçilmemiş tool
- ⚠️ Sarı tooltip: Düşük confidence uyarısı

**Props:**
- `toolSelection`: Tool seçim reasoning array'i

**Hover Davranışı:**
- Tool üzerine gelince seçim nedeni ve faktörler gösterilir
- Düşük confidence'ta uyarı tooltip'i görünür

---

#### CategoryMatchingChart
**Dosya:** `CategoryMatchingChart.stories.tsx`

Kategori eşleştirme skorlarını bar chart ile gösterir.

**Öne Çıkan Stories:**
- `Default`: Karışık skorlar
- `HighScoresOnly`: Sadece yüksek skorlar
- `LowScoresOnly`: Sadece düşük skorlar
- `ManyCategories`: Çok sayıda kategori (scroll)
- `WithClickHandler`: Tıklanabilir kategoriler

**Renk Kodlaması:**
- 🟢 Yeşil: Skor >0.7
- 🟡 Sarı: Skor 0.3-0.7
- 🔴 Kırmızı: Skor <0.3

**Props:**
- `categories`: Kategori eşleştirme array'i
- `onCategoryClick`: Kategori tıklama callback'i (opsiyonel)

**İnteraktif Özellikler:**
- Kategoriye tıklayınca eşleştirme nedenleri gösterilir
- Skorlar yüzde olarak formatlanır

---

### 3. Attention Weights ve Thinking Steps

#### AttentionWeightsChart
**Dosya:** `AttentionWeightsChart.stories.tsx`

Model attention ağırlıklarını bar chart veya radar chart ile gösterir.

**Öne Çıkan Stories:**
- `BarChart`: Bar chart görünümü
- `RadarChart`: Radar chart görünümü
- `HighHobbyWeight`: Hobi ağırlığı yüksek
- `BalancedWeights`: Dengeli ağırlıklar

**Chart Tipleri:**
- 📊 Bar Chart: Karşılaştırma için ideal
- 🎯 Radar Chart: Genel dağılımı görmek için

**Props:**
- `attentionWeights`: User ve gift feature ağırlıkları
- `chartType`: 'bar' veya 'radar'
- `onChartTypeChange`: Chart tipi değiştirme callback'i

**Özellikler:**
- Ağırlıklar yüzde olarak gösterilir
- Hover'da tam değer tooltip'te görünür
- Chart tipi toggle butonu ile değiştirilebilir

---

#### ThinkingStepsTimeline
**Dosya:** `ThinkingStepsTimeline.stories.tsx`

Modelin düşünme adımlarını kronolojik timeline'da gösterir.

**Öne Çıkan Stories:**
- `Default`: 5 adımlı standart timeline
- `SingleStep`: Tek adım
- `LongTimeline`: 10+ adım (scroll)
- `WithClickHandler`: Tıklanabilir adımlar
- `OutOfOrderSteps`: Otomatik sıralama örneği

**Timeline Özellikleri:**
- ✅ Yeşil checkmark: Tamamlanmış adım
- 📍 Adım numarası, action, result, insight
- 🔽 Tıklayınca detaylar genişler
- ⌨️ Klavye navigasyonu (Tab, Enter, Space)

**Props:**
- `steps`: Thinking step array'i
- `onStepClick`: Adım tıklama callback'i (opsiyonel)

---

### 4. Panel ve Kompozit Bileşenler

#### ReasoningPanel
**Dosya:** `ReasoningPanel.stories.tsx`

Tüm reasoning bilgilerini içeren detaylı panel.

**Öne Çıkan Stories:**
- `Default`: Tüm bölümler açık
- `ToolSelectionOnly`: Sadece tool selection
- `CategoryMatchingOnly`: Sadece category matching
- `AttentionWeightsOnly`: Sadece attention weights
- `ThinkingStepsOnly`: Sadece thinking steps
- `MobileView`: Mobil görünüm

**Filtre Seçenekleri:**
- Tool Selection
- Category Matching
- Attention Weights
- Thinking Steps

**Props:**
- `isOpen`: Panel açık/kapalı
- `onClose`: Panel kapatma callback'i
- `reasoningTrace`: Tüm reasoning bilgileri
- `gift`: Hediye bilgileri
- `userProfile`: Kullanıcı profili
- `activeFilters`: Aktif filtreler
- `onFilterChange`: Filtre değiştirme callback'i

**Responsive Davranış:**
- Desktop: Yan panel
- Mobile: Full-screen modal
- Swipe gesture ile kapatma (mobil)

---

#### ComparisonView
**Dosya:** `ComparisonView.stories.tsx`

Seçili hediyeleri yan yana karşılaştırır.

**Öne Çıkan Stories:**
- `TwoGifts`: İki hediye karşılaştırma
- `ThreeGifts`: Üç hediye karşılaştırma
- `MixedConfidence`: Farklı güven skorları
- `DifferentPriceRanges`: Farklı fiyat aralıkları
- `MobileView`: Mobil görünüm

**Karşılaştırma Özellikleri:**
- 🎴 Yan yana hediye kartları
- 📊 Kategori skorları karşılaştırma chart'ı
- 🎯 Attention weights overlay
- 📋 Güven skoru karşılaştırma tablosu

**Props:**
- `recommendations`: Karşılaştırılacak hediyeler
- `onExit`: Karşılaştırma modundan çıkış callback'i

---

### 5. Loading ve Error States

#### LoadingStates
**Dosya:** `LoadingStates.stories.tsx`

Yükleme durumları için skeleton loader'lar ve spinner'lar.

**Bileşenler:**
- `Spinner`: Küçük, orta, büyük boyutlarda
- `GiftCardSkeleton`: Hediye kartı skeleton'u
- `ToolSelectionSkeleton`: Tool selection skeleton'u
- `CategoryChartSkeleton`: Kategori chart skeleton'u
- `AttentionWeightsSkeleton`: Attention weights skeleton'u
- `ThinkingStepsSkeleton`: Thinking steps skeleton'u
- `ReasoningPanelSkeleton`: Tüm panel skeleton'u
- `LoadingOverlay`: Full-screen loading overlay

**Kullanım:**
```tsx
{isLoading ? <GiftCardSkeleton /> : <GiftRecommendationCard {...props} />}
```

---

#### ErrorStates
**Dosya:** `ErrorStates.stories.tsx`

Hata durumları için error message bileşenleri.

**Bileşenler:**
- `ErrorMessage`: Genel hata mesajı (retry ile)
- `InlineErrorMessage`: Inline hata mesajı
- `ReasoningUnavailableError`: Reasoning mevcut değil
- `NetworkError`: Ağ hatası
- `TimeoutError`: Timeout hatası
- `EmptyState`: Boş durum mesajı

**Öne Çıkan Stories:**
- `BasicError`: Standart hata + retry
- `ErrorWithoutRetry`: Retry olmadan hata
- `NetworkErrorStory`: Ağ bağlantı hatası
- `EmptyStateWithAction`: Boş durum + aksiyon butonu

---

### 6. Form Bileşenleri

#### UserProfileForm
**Dosya:** `UserProfileForm.stories.tsx`

Kullanıcı profili form bileşeni.

**Öne Çıkan Stories:**
- `Default`: Boş form
- `Loading`: Yükleme durumu
- `WithInitialValues`: Dolu form
- `MinimalBudget`: Düşük bütçe
- `HighBudget`: Yüksek bütçe

**Props:**
- `onSubmit`: Form submit callback'i
- `isLoading`: Yükleme durumu
- `initialValues`: Başlangıç değerleri (opsiyonel)

---

#### RecommendationCard
**Dosya:** `RecommendationCard.stories.tsx`

Basit hediye öneri kartı (reasoning olmadan).

**Öne Çıkan Stories:**
- `Default`: Standart kart
- `HighConfidence`: Yüksek güven
- `OutOfStock`: Stokta yok
- `ExpensiveItem`: Pahalı ürün
- `LowRating`: Düşük rating

---

## Erişilebilirlik (a11y)

Tüm bileşenler Storybook'un `@storybook/addon-a11y` eklentisi ile test edilmiştir.

### Erişilebilirlik Özellikleri:
- ✅ ARIA labels ve roles
- ✅ Klavye navigasyonu (Tab, Enter, Space, Arrow keys)
- ✅ Screen reader uyumluluğu
- ✅ Renk kontrast oranları (WCAG AA)
- ✅ Renk körlüğü dostu (icon + renk kombinasyonu)
- ✅ Focus management
- ✅ Semantic HTML

### Test Etme:
1. Storybook'ta bir story açın
2. Alt panelde "Accessibility" sekmesine tıklayın
3. Violations ve passes listesini inceleyin

---

## Responsive Tasarım

Tüm bileşenler responsive olarak tasarlanmıştır.

### Breakpoint'ler:
- **Mobile**: <768px
- **Tablet**: 768px - 1024px
- **Desktop**: >1024px

### Responsive Davranışlar:
- Chart'lar mobilde dikey layout'a geçer
- Panel'ler mobilde full-screen modal olur
- Touch gesture'lar mobilde desteklenir
- Tooltip'ler mobilde touch-friendly

### Test Etme:
Storybook'ta viewport değiştirmek için:
1. Toolbar'da viewport seçiciyi kullanın
2. Veya story'de `parameters.viewport` ayarlayın

---

## Tema Desteği

Bileşenler light ve dark tema destekler.

### Tema Değiştirme:
Storybook toolbar'ında background seçiciyi kullanın:
- ☀️ Light theme (varsayılan)
- 🌙 Dark theme

### Tailwind Dark Mode:
```tsx
className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
```

---

## Best Practices

### Story Yazarken:
1. **Açıklayıcı isimler kullanın**: `HighConfidence`, `LongReasoning`
2. **JSDoc yorumları ekleyin**: Story'nin ne gösterdiğini açıklayın
3. **Args kullanın**: Interaktif kontroller için
4. **Actions kullanın**: Callback'leri test etmek için
5. **Variants oluşturun**: Farklı durumları gösterin

### Dokümantasyon:
1. **Component description**: Meta'da açıklama ekleyin
2. **Props documentation**: ArgTypes ile props'ları belgeleyin
3. **Usage examples**: Story'lerde kullanım örnekleri gösterin
4. **Accessibility notes**: Erişilebilirlik özelliklerini belirtin

### Test Coverage:
Her bileşen için şu story'leri oluşturun:
- ✅ Default state
- ✅ Loading state
- ✅ Error state
- ✅ Empty state
- ✅ Edge cases
- ✅ Interactive states
- ✅ Responsive variants

---

## Storybook Eklentileri

### Yüklü Eklentiler:
- `@storybook/addon-essentials`: Temel eklentiler (controls, actions, docs)
- `@storybook/addon-interactions`: Interaction testing
- `@storybook/addon-a11y`: Erişilebilirlik testi
- `@storybook/addon-links`: Story'ler arası linkler

### Kullanım:
- **Controls**: Props'ları interaktif olarak değiştirin
- **Actions**: Callback'lerin çağrıldığını görün
- **Docs**: Auto-generated dokümantasyon
- **Accessibility**: a11y violations'ları görün

---

## Geliştirme İpuçları

### Yeni Story Ekleme:
```tsx
export const YourStoryName: Story = {
  args: {
    // props here
  },
  parameters: {
    // story-specific parameters
  },
};
```

### Mock Data Oluşturma:
```tsx
const mockData = {
  // realistic test data
};
```

### Interaktif Story:
```tsx
const InteractiveWrapper = (args) => {
  const [state, setState] = useState(initialState);
  return <Component {...args} state={state} onChange={setState} />;
};

export const Interactive: Story = {
  render: (args) => <InteractiveWrapper {...args} />,
};
```

---

## Sorun Giderme

### Story görünmüyor:
- Dosya adının `.stories.tsx` ile bittiğinden emin olun
- `meta` export'unun doğru olduğunu kontrol edin
- Storybook'u yeniden başlatın

### Props çalışmıyor:
- `argTypes` tanımlarını kontrol edin
- TypeScript type'larının doğru olduğundan emin olun

### Stil sorunları:
- Tailwind CSS'in yüklendiğinden emin olun
- `.storybook/preview.ts`'de `index.css` import edilmiş mi kontrol edin

---

## Kaynaklar

- [Storybook Dokümantasyonu](https://storybook.js.org/docs/react/get-started/introduction)
- [Recharts Dokümantasyonu](https://recharts.org/en-US/)
- [Radix UI Dokümantasyonu](https://www.radix-ui.com/docs/primitives/overview/introduction)
- [Tailwind CSS Dokümantasyonu](https://tailwindcss.com/docs)

---

## Katkıda Bulunma

Yeni story eklerken:
1. Bileşenin tüm state'lerini kapsayın
2. Erişilebilirlik özelliklerini test edin
3. Responsive davranışı kontrol edin
4. JSDoc yorumları ekleyin
5. README'yi güncelleyin

---

**Son Güncelleme:** 2024
**Versiyon:** 1.0.0
