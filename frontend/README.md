# Trendyol Gift Recommendation - Frontend

React + TypeScript + Vite tabanlı modern web uygulaması.

## Teknoloji Stack

- **React 18+**: UI framework
- **TypeScript 5+**: Type safety
- **Vite**: Build tool ve dev server
- **React Router**: Routing
- **Zustand**: State management
- **TanStack Query**: Data fetching ve caching
- **Tailwind CSS**: Styling
- **Axios**: HTTP client
- **Recharts**: Data visualization
- **Vitest**: Testing framework
- **fast-check**: Property-based testing

## Proje Yapısı

```
frontend/
├── src/
│   ├── lib/
│   │   ├── api/           # API client ve type definitions
│   │   │   ├── client.ts
│   │   │   ├── types.ts
│   │   │   └── recommendations.ts
│   │   └── utils/         # Utility fonksiyonlar
│   │       └── cn.ts
│   ├── hooks/             # Custom React hooks
│   │   ├── useHealth.ts
│   │   └── useRecommendations.ts
│   ├── store/             # Zustand store
│   │   └── useAppStore.ts
│   ├── pages/             # Page components
│   │   └── HomePage.tsx
│   ├── routes/            # Route definitions
│   │   └── index.tsx
│   ├── test/              # Test setup
│   │   └── setup.ts
│   ├── App.tsx            # Root component
│   ├── main.tsx           # Entry point
│   └── index.css          # Global styles
├── public/                # Static assets
├── .env.example           # Environment variables template
├── vite.config.ts         # Vite configuration
├── tailwind.config.js     # Tailwind configuration
├── tsconfig.json          # TypeScript configuration
└── package.json           # Dependencies
```

## Kurulum

```bash
npm install
```

## Geliştirme

```bash
npm run dev
```

Uygulama http://localhost:3000 adresinde çalışacaktır.

## Build

```bash
npm run build
```

## Test

```bash
# Tüm testleri çalıştır (Unit + Property tests)
npm run test

# Watch mode
npm run test:watch

# Coverage raporu
npm run test:coverage

# Property-based tests
npm run test:property

# E2E tests (Playwright)
npm run test:e2e

# E2E tests with UI
npm run test:e2e:ui
```

### Test İstatistikleri
- **Total Tests**: 667 passing
- **Property Tests**: 50+ with 100+ iterations each
- **E2E Tests**: 5 comprehensive flows
- **Coverage**: 85%+

## Linting

```bash
# Lint kontrolü
npm run lint

# Lint düzeltme
npm run lint:fix
```

## Type Checking

```bash
npm run type-check
```

## Özellikler

### ✅ Tamamlanan

#### Core Infrastructure
- [x] Vite + React + TypeScript projesi
- [x] Tailwind CSS konfigürasyonu
- [x] React Router setup
- [x] Zustand state management
- [x] TanStack Query setup
- [x] Axios API client
- [x] Type definitions
- [x] Test altyapısı (Vitest + Playwright)
- [x] Property-based testing (fast-check)
- [x] Linting ve formatting
- [x] Storybook documentation

#### Reasoning Visualization Features
- [x] Gift Recommendation Cards with Reasoning
- [x] Confidence Indicators (High/Medium/Low)
- [x] Tool Selection Visualization
- [x] Category Matching Charts
- [x] Attention Weights Charts (Bar & Radar)
- [x] Thinking Steps Timeline
- [x] Detailed Reasoning Panel
- [x] Comparison Mode (Side-by-side)
- [x] Export Functionality (JSON, PDF, Share)
- [x] Responsive Design (Mobile, Tablet, Desktop)
- [x] Loading & Error States
- [x] Accessibility (WCAG AA)

#### Performance Optimizations
- [x] Lazy Loading
- [x] Code Splitting
- [x] React.memo optimization
- [x] Virtual Scrolling
- [x] Bundle size optimization (196KB)

### 🚧 Gelecek Görevler

- [ ] Dark mode implementation
- [ ] i18n support (multi-language)
- [ ] Reasoning history
- [ ] Advanced analytics dashboard
- [ ] Custom reasoning filters
- [ ] Excel/CSV export

## Environment Variables

`.env.example` dosyasını `.env` olarak kopyalayın:

```bash
cp .env.example .env
```

Gerekli değişkenler:

- `VITE_API_BASE_URL`: Backend API URL (default: `/api`)

## API Integration

Backend API ile iletişim için `@tanstack/react-query` kullanılmaktadır:

```typescript
import { useRecommendations } from '@/hooks/useRecommendations';

function MyComponent() {
  const { mutate, data, isLoading } = useRecommendations();
  
  const handleSubmit = (profile: UserProfile) => {
    mutate({ userProfile: profile });
  };
  
  // ...
}
```

## State Management

Global state için Zustand kullanılmaktadır:

```typescript
import { useAppStore } from '@/store/useAppStore';

function MyComponent() {
  const { theme, toggleTheme } = useAppStore();
  
  // ...
}
```

## Styling

Tailwind CSS utility-first yaklaşımı kullanılmaktadır. Custom class merge için `cn` utility fonksiyonu mevcuttur:

```typescript
import { cn } from '@/lib/utils/cn';

<div className={cn('base-class', isActive && 'active-class')} />
```

## Storybook

Component documentation ve interactive playground:

```bash
# Storybook başlat
npm run storybook

# Storybook build
npm run build-storybook
```

Storybook http://localhost:6006 adresinde çalışacaktır.

## Docker Deployment

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## Performance

- **Bundle Size**: 196KB (minified)
- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <3s
- **Lighthouse Score**: 90+

## Browser Support

- Chrome (son 2 versiyon)
- Firefox (son 2 versiyon)
- Safari (son 2 versiyon)
- Edge (son 2 versiyon)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Documentation

- [Integration Summary](./INTEGRATION_SUMMARY.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- [Reasoning Setup Guide](./REASONING_SETUP.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)

## Contributing

1. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
2. Değişiklikleri commit et (`git commit -m 'Add amazing feature'`)
3. Branch'i push et (`git push origin feature/amazing-feature`)
4. Pull Request aç

## License

MIT License - detaylar için [LICENSE](../LICENSE) dosyasına bakın.
