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
# Tüm testleri çalıştır
npm run test

# Watch mode
npm run test:watch

# Coverage raporu
npm run test:coverage
```

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

- [x] Vite + React + TypeScript projesi
- [x] Tailwind CSS konfigürasyonu
- [x] React Router setup
- [x] Zustand state management
- [x] TanStack Query setup
- [x] Axios API client
- [x] Type definitions
- [x] Test altyapısı
- [x] Linting ve formatting

### 🚧 Gelecek Görevler

- [ ] UserProfileForm component
- [ ] RecommendationCard component
- [ ] ToolResultsModal component
- [ ] Theme switching (dark mode)
- [ ] Responsive design
- [ ] Error handling
- [ ] Loading states

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
