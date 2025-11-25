/**
 * Optimized components for better performance
 * 
 * This module exports performance-optimized versions of reasoning components:
 * - Lazy-loaded components for code splitting
 * - Memoized components to prevent unnecessary re-renders
 * - Virtual scrolling for long lists
 * - Optimized chart rendering with debouncing
 * 
 * Usage:
 * Import from this module instead of the original components when performance is critical
 * 
 * @example
 * ```tsx
 * import { 
 *   LazyReasoningPanel, 
 *   MemoizedGiftRecommendationCard,
 *   VirtualThinkingStepsTimeline 
 * } from '@/components/optimized';
 * ```
 */

// Lazy-loaded components
export { LazyReasoningPanel } from '../LazyReasoningPanel';
export { 
  LazyAttentionWeightsChart, 
  LazyCategoryMatchingChart 
} from '../LazyCharts';

// Memoized components
export {
  MemoizedToolSelectionCard,
  MemoizedThinkingStepsTimeline,
  MemoizedConfidenceIndicator,
  MemoizedGiftRecommendationCard,
} from '../MemoizedComponents';

// Virtual scrolling components
export { VirtualThinkingStepsTimeline } from '../VirtualThinkingStepsTimeline';

// Optimized chart components
export { OptimizedAttentionWeightsChart } from '../OptimizedAttentionWeightsChart';

// Re-export hooks
export { 
  useOptimizedChart, 
  useThrottle, 
  useInViewport 
} from '../../hooks/useOptimizedChart';

// Re-export code splitting utilities
export {
  preloadComponent,
  lazyWithRetry,
  prefetchComponents,
  lazyWithTimeout,
} from '../../lib/utils/codeSplitting';
