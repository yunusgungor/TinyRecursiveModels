import { lazy, Suspense } from 'react';
import { LoadingSkeleton } from './LoadingStates';
import type { AttentionWeightsChartProps } from './AttentionWeightsChart';
import type { CategoryMatchingChartProps } from './CategoryMatchingChart';

/**
 * Lazy-loaded chart components
 * Reduces initial bundle size by code-splitting Recharts library
 */

const AttentionWeightsChartLazy = lazy(() =>
  import('./AttentionWeightsChart').then(module => ({ default: module.AttentionWeightsChart }))
);

const CategoryMatchingChartLazy = lazy(() =>
  import('./CategoryMatchingChart').then(module => ({ default: module.CategoryMatchingChart }))
);

export const LazyAttentionWeightsChart: React.FC<AttentionWeightsChartProps> = (props) => {
  return (
    <Suspense fallback={<LoadingSkeleton variant="chart" />}>
      <AttentionWeightsChartLazy {...props} />
    </Suspense>
  );
};

export const LazyCategoryMatchingChart: React.FC<CategoryMatchingChartProps> = (props) => {
  return (
    <Suspense fallback={<LoadingSkeleton variant="chart" />}>
      <CategoryMatchingChartLazy {...props} />
    </Suspense>
  );
};
