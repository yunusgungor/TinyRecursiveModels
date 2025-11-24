# Reasoning Visualization Setup

This document describes the setup completed for the frontend reasoning visualization feature.

## Dependencies Installed

### UI Components
- `@radix-ui/react-dialog` - Modal dialogs for reasoning panels
- `@radix-ui/react-tooltip` - Tooltips for tool information
- `@radix-ui/react-tabs` - Tab navigation for reasoning sections
- `@radix-ui/react-select` - Select dropdowns for filters
- `@radix-ui/react-switch` - Toggle switches for settings

### Charts
- `recharts` - Already installed, used for bar charts and radar charts

### PDF Export
- `jspdf` - PDF generation for reasoning export

### Testing
- `fast-check` - Already installed, property-based testing library
- `@playwright/test` - E2E testing framework

## Configuration Files

### TypeScript Types
- `src/types/reasoning.ts` - Complete type definitions for reasoning data models
- `src/types/index.ts` - Central export for all types
- `src/vite-env.d.ts` - Updated with reasoning environment variables

### Configuration
- `src/config/featureFlags.ts` - Feature flag management for reasoning functionality
- `src/config/api.ts` - API configuration for reasoning endpoints

### Testing
- `src/test/propertyTestHelpers.ts` - Property-based testing generators and utilities
- `playwright.config.ts` - Playwright E2E testing configuration
- `vite.config.ts` - Updated with property-based testing configuration

### Environment
- `.env.example` - Updated with reasoning-related environment variables

## Environment Variables

The following environment variables are now available:

```bash
# Reasoning Feature Configuration
VITE_ENABLE_REASONING=true
VITE_DEFAULT_REASONING_LEVEL=basic
VITE_ENABLE_REASONING_EXPORT=true
VITE_ENABLE_REASONING_COMPARISON=true
VITE_MAX_THINKING_STEPS=20
VITE_REASONING_CACHE_TTL=300000
```

## Feature Flags

Feature flags can be accessed via:

```typescript
import { 
  isReasoningEnabled, 
  isReasoningExportEnabled,
  isReasoningComparisonEnabled,
  getDefaultReasoningLevel 
} from '@/config/featureFlags';
```

## Type Definitions

All reasoning types are available from:

```typescript
import {
  ToolSelectionReasoning,
  CategoryMatchingReasoning,
  AttentionWeights,
  ThinkingStep,
  ConfidenceExplanation,
  ReasoningTrace,
  EnhancedGiftRecommendation,
  // ... and more
} from '@/types/reasoning';
```

## Property-Based Testing

Property-based testing helpers are available:

```typescript
import {
  arbToolSelectionReasoning,
  arbCategoryMatchingReasoning,
  arbAttentionWeights,
  arbReasoningTrace,
  runPropertyTest,
  MIN_PBT_ITERATIONS, // 100 iterations minimum
} from '@/test/propertyTestHelpers';
```

## NPM Scripts

New scripts added:

```bash
# Run property-based tests only
npm run test:property

# Run E2E tests
npm run test:e2e

# Run E2E tests in UI mode
npm run test:e2e:ui

# Run E2E tests in debug mode
npm run test:e2e:debug
```

## Next Steps

The project is now ready for implementing reasoning visualization components. The next tasks in the implementation plan are:

1. Core Data Models and Types ✅ (Completed in this task)
2. API Service Layer
3. Custom Hooks Implementation
4. Component Development

All dependencies are installed, types are defined, and testing infrastructure is configured.
