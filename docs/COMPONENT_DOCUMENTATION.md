# Frontend Component Documentation

Bu doküman, Trendyol Gift Recommendation uygulamasının frontend bileşenlerini detaylı olarak açıklar.

## İçindekiler

1. [Storybook Kullanımı](#storybook-kullanımı)
2. [Core Components](#core-components)
3. [Layout Components](#layout-components)
4. [Form Components](#form-components)
5. [Display Components](#display-components)
6. [Utility Components](#utility-components)

## Storybook Kullanımı

### Storybook'u Başlatma

```bash
cd frontend
npm run storybook
```

Storybook, `http://localhost:6006` adresinde açılacaktır.

### Storybook Build

```bash
npm run build-storybook
```

Build edilen dosyalar `storybook-static/` klasöründe oluşturulur.

## Core Components

### UserProfileForm

Kullanıcı profil bilgilerini toplamak için kullanılan ana form bileşeni.

**Import:**
```typescript
import { UserProfileForm } from '@/components/UserProfileForm';
```

**Props:**
```typescript
interface UserProfileFormProps {
  onSubmit: (profile: UserProfile) => void;
  isLoading: boolean;
  initialValues?: Partial<UserProfile>;
}
```

**Usage:**
```typescript
const MyComponent = () => {
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async (profile: UserProfile) => {
    setIsLoading(true);
    try {
      await submitProfile(profile);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <UserProfileForm 
      onSubmit={handleSubmit}
      isLoading={isLoading}
    />
  );
};
```

**Features:**
- Real-time validation
- Multi-select hobbies
- Budget formatting (Turkish Lira)
- Age validation (18-100)
- Personality traits selection
- Responsive design

**Validation Rules:**
- Age: 18-100 (required)
- Hobbies: 1-10 items (required)
- Relationship: non-empty string (required)
- Budget: positive number (required)
- Occasion: non-empty string (required)
- Personality traits: 0-5 items (optional)

**Accessibility:**
- ARIA labels on all inputs
- Keyboard navigation support
- Screen reader friendly
- Focus management

### RecommendationCard

Ürün önerilerini görselleştiren kart bileşeni.

**Import:**
```typescript
import { RecommendationCard } from '@/components/RecommendationCard';
```

**Props:**
```typescript
interface RecommendationCardProps {
  gift: GiftItem;
  toolResults: ToolResults;
  confidenceScore: number;
  reasoning: string[];
  rank: number;
  onDetailsClick: () => void;
  onTrendyolClick: () => void;
}
```

**Usage:**
```typescript
<RecommendationCard
  gift={giftItem}
  toolResults={toolResults}
  confidenceScore={0.92}
  reasoning={[
    'Matches user\'s cooking hobby',
    'Within budget range'
  ]}
  rank={1}
  onDetailsClick={() => setShowModal(true)}
  onTrendyolClick={() => window.open(giftItem.trendyol_url)}
/>
```

**Features:**
- Product image with lazy loading
- Price display (formatted)
- Rating stars
- Confidence score indicator
- Stock status badge
- Responsive layout
- Hover effects

**Confidence Score Colors:**
- 0.8-1.0: Green (Highly recommended)
- 0.5-0.8: Yellow (Recommended)
- 0.0-0.5: Red (Low confidence)

**Accessibility:**
- Alt text for images
- ARIA labels for buttons
- Keyboard accessible
- Color contrast compliant

### ToolResultsModal

Tool analiz sonuçlarını detaylı olarak gösteren modal bileşeni.

**Import:**
```typescript
import { ToolResultsModal } from '@/components/ToolResultsModal';
```

**Props:**
```typescript
interface ToolResultsModalProps {
  gift: GiftItem;
  toolResults: ToolResults;
  isOpen: boolean;
  onClose: () => void;
}
```

**Usage:**
```typescript
const [isOpen, setIsOpen] = useState(false);

<ToolResultsModal
  gift={selectedGift}
  toolResults={toolResults}
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
/>
```

**Features:**
- Price comparison chart
- Review sentiment analysis
- Trend visualization
- Budget breakdown
- Responsive modal
- Close on escape key
- Click outside to close

**Sections:**
1. **Price Comparison**
   - Best price
   - Average price
   - Savings percentage
   - Compared platforms

2. **Review Analysis**
   - Average rating
   - Total reviews
   - Sentiment score
   - Key positives/negatives

3. **Trend Analysis**
   - Popularity score
   - Trend direction
   - Growth rate
   - Related trending items

4. **Budget Optimizer**
   - Recommended allocation
   - Value score
   - Savings opportunities

### ThemeToggle

Karanlık/aydınlık tema değiştirme bileşeni.

**Import:**
```typescript
import { ThemeToggle } from '@/components/ThemeToggle';
```

**Props:**
```typescript
interface ThemeToggleProps {
  className?: string;
}
```

**Usage:**
```typescript
<ThemeToggle className="ml-auto" />
```

**Features:**
- Smooth theme transition
- Persists to localStorage
- Icon animation
- Accessible button

### LazyImage

Lazy loading destekli görsel bileşeni.

**Import:**
```typescript
import { LazyImage } from '@/components/LazyImage';
```

**Props:**
```typescript
interface LazyImageProps {
  src: string;
  alt: string;
  className?: string;
  fallback?: string;
}
```

**Usage:**
```typescript
<LazyImage
  src={product.image_url}
  alt={product.name}
  className="w-full h-48 object-cover"
  fallback="/placeholder.png"
/>
```

**Features:**
- Intersection Observer API
- Blur-up effect
- Error fallback
- Responsive images

## Layout Components

### HomePage

Ana sayfa layout bileşeni.

**Location:** `src/pages/HomePage.tsx`

**Features:**
- Header with navigation
- Main content area
- Footer
- Responsive grid layout

**Sections:**
1. Hero section with form
2. Recommendations grid
3. Favorites sidebar
4. Search history

### Header

Uygulama başlığı ve navigasyon.

**Features:**
- Logo
- Navigation links
- Theme toggle
- Mobile menu

### Footer

Uygulama alt bilgi bölümü.

**Features:**
- Copyright
- Links
- Social media icons

## Form Components

### Input

Temel input bileşeni.

**Props:**
```typescript
interface InputProps {
  label: string;
  type?: string;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  required?: boolean;
  placeholder?: string;
}
```

**Usage:**
```typescript
<Input
  label="Yaş"
  type="number"
  value={age}
  onChange={setAge}
  error={ageError}
  required
/>
```

### Select

Dropdown seçim bileşeni.

**Props:**
```typescript
interface SelectProps {
  label: string;
  options: Array<{ value: string; label: string }>;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  required?: boolean;
}
```

**Usage:**
```typescript
<Select
  label="İlişki Durumu"
  options={relationshipOptions}
  value={relationship}
  onChange={setRelationship}
  required
/>
```

### MultiSelect

Çoklu seçim bileşeni.

**Props:**
```typescript
interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  max?: number;
  error?: string;
}
```

**Usage:**
```typescript
<MultiSelect
  label="Hobiler"
  options={hobbyOptions}
  selected={hobbies}
  onChange={setHobbies}
  max={10}
/>
```

## Display Components

### Card

Genel amaçlı kart bileşeni.

**Props:**
```typescript
interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}
```

**Usage:**
```typescript
<Card className="p-4">
  <h3>Title</h3>
  <p>Content</p>
</Card>
```

### Badge

Etiket/rozet bileşeni.

**Props:**
```typescript
interface BadgeProps {
  children: React.ReactNode;
  variant?: 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
}
```

**Usage:**
```typescript
<Badge variant="success">Stokta</Badge>
<Badge variant="error">Stokta Yok</Badge>
```

### Rating

Yıldız değerlendirme bileşeni.

**Props:**
```typescript
interface RatingProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  showValue?: boolean;
}
```

**Usage:**
```typescript
<Rating value={4.5} max={5} showValue />
```

### ProgressBar

İlerleme çubuğu bileşeni.

**Props:**
```typescript
interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  color?: string;
}
```

**Usage:**
```typescript
<ProgressBar 
  value={75} 
  max={100} 
  label="Güven Skoru"
  color="green"
/>
```

## Utility Components

### Spinner

Yükleme animasyonu bileşeni.

**Props:**
```typescript
interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
}
```

**Usage:**
```typescript
<Spinner size="lg" />
```

### ErrorBoundary

Hata yakalama bileşeni.

**Props:**
```typescript
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}
```

**Usage:**
```typescript
<ErrorBoundary fallback={<ErrorFallback />}>
  <App />
</ErrorBoundary>
```

### Portal

React Portal bileşeni.

**Props:**
```typescript
interface PortalProps {
  children: React.ReactNode;
  container?: HTMLElement;
}
```

**Usage:**
```typescript
<Portal>
  <Modal />
</Portal>
```

## Styling

### Tailwind CSS Classes

**Common Patterns:**

```typescript
// Container
"container mx-auto px-4"

// Card
"bg-white dark:bg-gray-800 rounded-lg shadow-md p-6"

// Button Primary
"bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"

// Button Secondary
"bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-4 rounded"

// Input
"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"

// Error Text
"text-red-600 text-sm mt-1"
```

### Dark Mode

Tüm bileşenler dark mode destekler:

```typescript
// Light mode
"bg-white text-gray-900"

// Dark mode
"dark:bg-gray-800 dark:text-gray-100"
```

## Responsive Design

### Breakpoints

```typescript
// Tailwind breakpoints
sm: '640px'   // Mobile landscape
md: '768px'   // Tablet
lg: '1024px'  // Desktop
xl: '1280px'  // Large desktop
2xl: '1536px' // Extra large
```

### Responsive Patterns

```typescript
// Mobile first
"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3"

// Hide on mobile
"hidden md:block"

// Show only on mobile
"block md:hidden"

// Responsive padding
"p-4 md:p-6 lg:p-8"
```

## Testing Components

### Unit Tests

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { RecommendationCard } from './RecommendationCard';

describe('RecommendationCard', () => {
  it('renders product information', () => {
    render(<RecommendationCard {...mockProps} />);
    
    expect(screen.getByText('Premium Coffee Set')).toBeInTheDocument();
    expect(screen.getByText('299.99 ₺')).toBeInTheDocument();
  });
  
  it('calls onDetailsClick when details button clicked', () => {
    const handleClick = jest.fn();
    render(<RecommendationCard {...mockProps} onDetailsClick={handleClick} />);
    
    fireEvent.click(screen.getByText('Detaylar'));
    expect(handleClick).toHaveBeenCalled();
  });
});
```

### Storybook Tests

```typescript
import { composeStories } from '@storybook/react';
import * as stories from './RecommendationCard.stories';

const { Default, LowConfidence } = composeStories(stories);

describe('RecommendationCard Stories', () => {
  it('renders default story', () => {
    const { container } = render(<Default />);
    expect(container).toMatchSnapshot();
  });
});
```

## Best Practices

### Component Structure

```typescript
// 1. Imports
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

// 2. Types
interface Props {
  // ...
}

// 3. Component
export const MyComponent: React.FC<Props> = ({ prop1, prop2 }) => {
  // 4. Hooks
  const [state, setState] = useState();
  const { data } = useQuery();
  
  // 5. Effects
  useEffect(() => {
    // ...
  }, []);
  
  // 6. Handlers
  const handleClick = () => {
    // ...
  };
  
  // 7. Render helpers
  const renderContent = () => {
    // ...
  };
  
  // 8. Main render
  return (
    <div>
      {renderContent()}
    </div>
  );
};
```

### Performance

```typescript
// Memoization
const MemoizedComponent = React.memo(MyComponent);

// useMemo for expensive calculations
const expensiveValue = useMemo(() => {
  return calculateExpensiveValue(data);
}, [data]);

// useCallback for functions
const handleClick = useCallback(() => {
  doSomething(id);
}, [id]);
```

### Accessibility

```typescript
// ARIA labels
<button aria-label="Close modal">
  <CloseIcon />
</button>

// Semantic HTML
<nav>
  <ul>
    <li><a href="/">Home</a></li>
  </ul>
</nav>

// Keyboard navigation
<div
  role="button"
  tabIndex={0}
  onKeyPress={handleKeyPress}
  onClick={handleClick}
>
  Click me
</div>
```

## Resources

- [Storybook Documentation](https://storybook.js.org/docs/react/get-started/introduction)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Testing Library](https://testing-library.com/docs/react-testing-library/intro/)

---

**Last Updated:** January 2024
