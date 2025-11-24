# Contexts

This directory contains React Context providers for global state management.

## ReasoningContext

The `ReasoningContext` provides centralized state management for all reasoning-related features in the application.

### Features

- **Reasoning Level Management**: Control the level of detail shown (basic/detailed/full)
- **Gift Selection**: Manage selected gifts for comparison mode
- **Comparison Mode**: Enable/disable comparison view
- **Panel State**: Control reasoning panel open/close state
- **Filters**: Manage which reasoning sections are visible
- **Chart Type**: Toggle between bar and radar charts
- **Persistence**: All preferences are automatically saved to localStorage

### Usage

#### 1. Wrap your app with the provider

```tsx
import { ReasoningProvider } from '@/contexts';

function App() {
  return (
    <ReasoningProvider>
      <YourApp />
    </ReasoningProvider>
  );
}
```

#### 2. Use the context in your components

```tsx
import { useReasoningContext } from '@/contexts';

function MyComponent() {
  const {
    reasoningLevel,
    setReasoningLevel,
    selectedGifts,
    toggleGiftSelection,
    isComparisonMode,
    setComparisonMode,
    isPanelOpen,
    openPanel,
    closePanel,
    activeFilters,
    setFilters,
    chartType,
    setChartType,
  } = useReasoningContext();

  return (
    <div>
      {/* Reasoning level selector */}
      <select value={reasoningLevel} onChange={(e) => setReasoningLevel(e.target.value)}>
        <option value="basic">Basic</option>
        <option value="detailed">Detailed</option>
        <option value="full">Full</option>
      </select>

      {/* Gift selection for comparison */}
      <button onClick={() => toggleGiftSelection('gift-123')}>
        Select Gift
      </button>

      {/* Comparison mode */}
      {selectedGifts.length >= 2 && (
        <button onClick={() => setComparisonMode(true)}>
          Compare {selectedGifts.length} Gifts
        </button>
      )}

      {/* Reasoning panel */}
      <button onClick={openPanel}>Show Details</button>
      
      {isPanelOpen && (
        <ReasoningPanel
          onClose={closePanel}
          activeFilters={activeFilters}
          chartType={chartType}
        />
      )}
    </div>
  );
}
```

### API Reference

#### State

- `reasoningLevel: 'basic' | 'detailed' | 'full'` - Current reasoning detail level
- `selectedGifts: string[]` - Array of selected gift IDs
- `isComparisonMode: boolean` - Whether comparison mode is active
- `isPanelOpen: boolean` - Whether reasoning panel is open
- `activeFilters: ReasoningFilter[]` - Active reasoning section filters
- `chartType: 'bar' | 'radar'` - Current chart visualization type

#### Actions

- `setReasoningLevel(level)` - Update reasoning level
- `toggleGiftSelection(giftId)` - Toggle gift selection for comparison
- `clearSelection()` - Clear all selected gifts and exit comparison mode
- `setComparisonMode(enabled)` - Enable/disable comparison mode
- `openPanel()` - Open reasoning panel
- `closePanel()` - Close reasoning panel
- `togglePanel()` - Toggle reasoning panel
- `setFilters(filters)` - Update active filters
- `setChartType(type)` - Change chart visualization type

### Persistence

The following preferences are automatically persisted to localStorage:

- Reasoning level (`reasoning-level`)
- Active filters (`reasoning-panel-filters`)
- Chart type (`reasoning-panel-chart-type`)

Gift selection and panel state are session-only (not persisted).

### Integration with Existing State

The `ReasoningContext` integrates with:

- `useReasoningLevel` hook for reasoning level management
- `useReasoningPanel` hook for panel state and preferences
- `useAppStore` (Zustand) for comparison mode and gift selection

This ensures consistency across the application and leverages existing state management patterns.

## ThemeContext

Provides theme management (light/dark mode) for the application.

See `ThemeContext.tsx` for implementation details.
