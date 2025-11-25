import React, { lazy, Suspense } from 'react';
import { LoadingSkeleton } from './LoadingStates';
import type { ReasoningPanelProps } from './ReasoningPanel';

/**
 * Lazy-loaded ReasoningPanel component
 * Reduces initial bundle size by code-splitting the heavy reasoning panel
 */
const ReasoningPanelLazy = lazy(() => 
  import('./ReasoningPanel').then(module => ({ default: module.ReasoningPanel }))
);

export const LazyReasoningPanel: React.FC<ReasoningPanelProps> = (props) => {
  return (
    <Suspense fallback={<LoadingSkeleton variant="panel" />}>
      <ReasoningPanelLazy {...props} />
    </Suspense>
  );
};
