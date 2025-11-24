/**
 * Loading state components for reasoning visualization
 * Provides skeleton loaders and spinners for various components
 */

import React from 'react';

/**
 * Generic spinner component for loading states
 */
export const Spinner: React.FC<{ size?: 'sm' | 'md' | 'lg'; className?: string }> = ({
  size = 'md',
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4',
  };

  return (
    <div
      className={`inline-block animate-spin rounded-full border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite] ${sizeClasses[size]} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
};

/**
 * Skeleton loader for gift recommendation cards
 */
export const GiftCardSkeleton: React.FC = () => {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 animate-pulse"
      role="status"
      aria-label="Loading gift recommendation"
    >
      {/* Image skeleton */}
      <div className="w-full h-48 bg-gray-300 dark:bg-gray-700 rounded-md mb-4" />

      {/* Title skeleton */}
      <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-3/4 mb-2" />

      {/* Price skeleton */}
      <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-1/4 mb-4" />

      {/* Reasoning lines skeleton */}
      <div className="space-y-2 mb-4">
        <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-full" />
        <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-5/6" />
        <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-4/6" />
      </div>

      {/* Confidence indicator skeleton */}
      <div className="h-8 bg-gray-300 dark:bg-gray-700 rounded w-1/3 mb-4" />

      {/* Button skeleton */}
      <div className="h-10 bg-gray-300 dark:bg-gray-700 rounded w-full" />
    </div>
  );
};

/**
 * Skeleton loader for tool selection card
 */
export const ToolSelectionSkeleton: React.FC = () => {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg p-4 animate-pulse"
      role="status"
      aria-label="Loading tool selection"
    >
      <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-1/2 mb-4" />
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex items-center space-x-3">
            <div className="w-5 h-5 bg-gray-300 dark:bg-gray-700 rounded" />
            <div className="flex-1 h-4 bg-gray-300 dark:bg-gray-700 rounded" />
            <div className="w-12 h-4 bg-gray-300 dark:bg-gray-700 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Skeleton loader for category matching chart
 */
export const CategoryChartSkeleton: React.FC = () => {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg p-4 animate-pulse"
      role="status"
      aria-label="Loading category chart"
    >
      <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-1/2 mb-4" />
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="space-y-2">
            <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-1/3" />
            <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-full" />
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Skeleton loader for attention weights chart
 */
export const AttentionWeightsSkeleton: React.FC = () => {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg p-4 animate-pulse"
      role="status"
      aria-label="Loading attention weights"
    >
      <div className="flex justify-between items-center mb-4">
        <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-1/3" />
        <div className="h-8 bg-gray-300 dark:bg-gray-700 rounded w-20" />
      </div>
      <div className="h-64 bg-gray-300 dark:bg-gray-700 rounded" />
    </div>
  );
};

/**
 * Skeleton loader for thinking steps timeline
 */
export const ThinkingStepsSkeleton: React.FC = () => {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg p-4 animate-pulse"
      role="status"
      aria-label="Loading thinking steps"
    >
      <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-1/2 mb-4" />
      <div className="space-y-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex space-x-3">
            <div className="w-8 h-8 bg-gray-300 dark:bg-gray-700 rounded-full flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-3/4" />
              <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-full" />
              <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-5/6" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Skeleton loader for reasoning panel
 */
export const ReasoningPanelSkeleton: React.FC = () => {
  return (
    <div
      className="space-y-6 animate-pulse"
      role="status"
      aria-label="Loading reasoning panel"
    >
      <ToolSelectionSkeleton />
      <CategoryChartSkeleton />
      <AttentionWeightsSkeleton />
      <ThinkingStepsSkeleton />
    </div>
  );
};

/**
 * Loading overlay for API requests
 */
export const LoadingOverlay: React.FC<{ message?: string }> = ({
  message = 'Yükleniyor...',
}) => {
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      role="status"
      aria-label={message}
    >
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 flex flex-col items-center space-y-4">
        <Spinner size="lg" className="text-blue-600" />
        <p className="text-gray-700 dark:text-gray-300 font-medium">{message}</p>
      </div>
    </div>
  );
};
