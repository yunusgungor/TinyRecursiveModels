# Trendyol Gift Recommendation - Frontend

React ve TypeScript ile geliştirilmiş modern web uygulaması.

## Özellikler

- React 18 ile modern UI
- TypeScript ile tip güvenliği
- Tailwind CSS ile responsive tasarım
- TanStack Query ile veri yönetimi
- Zustand ile state management
- Vitest ile test
- Dark mode desteği

## Kurulum

### Gereksinimler

- Node.js 20+
- npm veya yarn

### Yerel Geliştirme

```bash
# Bağımlılıkları yükle
npm install

# Environment variables ayarla
cp .env.example .env
# .env dosyasını düzenle

# Geliştirme sunucusunu başlat
npm run dev
```

Uygulama http://localhost:3000 adresinde çalışacaktır.

## Test

```bash
# Tüm testleri çalıştır
npm test

# Watch mode
npm run test:watch

# Coverage ile
npm run test:coverage
```

## Build

```bash
# Production build
npm run build

# Build'i önizle
npm run preview
```

## Proje Yapısı

```
frontend/
├── src/
│   ├── components/          # React bileşenleri
│   ├── pages/               # Sayfa bileşenleri
│   ├── hooks/               # Custom hooks
│   ├── services/            # API servisleri
│   ├── store/               # Zustand store
│   ├── types/               # TypeScript tipleri
│   ├── utils/               # Yardımcı fonksiyonlar
│   └── test/                # Test utilities
├── public/                  # Static dosyalar
└── index.html              # HTML template
```

## Kod Standartları

```bash
# Linting
npm run lint

# Linting + fix
npm run lint:fix

# Type checking
npm run type-check

# Formatting
npm run format
```
