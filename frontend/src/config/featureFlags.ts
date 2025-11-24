/// <reference types="../vite-env.d.ts" />

/**
 * Feature flags configuration for reasoning functionality
 * Allows enabling/disabling features based on environment variables
 */

export interface FeatureFlags {
  enableReasoning: boolean;
  defaultReasoningLevel: 'basic' | 'detailed' | 'full';
  enableReasoningExport: boolean;
  enableReasoningComparison: boolean;
  maxThinkingSteps: number;
  reasoningCacheTTL: number;
}

/**
 * Get feature flags from environment variables
 */
export const getFeatureFlags = (): FeatureFlags => {
  return {
    enableReasoning: import.meta.env.VITE_ENABLE_REASONING === 'true',
    defaultReasoningLevel: (import.meta.env.VITE_DEFAULT_REASONING_LEVEL || 'basic') as 'basic' | 'detailed' | 'full',
    enableReasoningExport: import.meta.env.VITE_ENABLE_REASONING_EXPORT === 'true',
    enableReasoningComparison: import.meta.env.VITE_ENABLE_REASONING_COMPARISON === 'true',
    maxThinkingSteps: parseInt(import.meta.env.VITE_MAX_THINKING_STEPS || '20', 10),
    reasoningCacheTTL: parseInt(import.meta.env.VITE_REASONING_CACHE_TTL || '300000', 10),
  };
};

/**
 * Feature flags instance
 */
export const featureFlags = getFeatureFlags();

/**
 * Check if reasoning feature is enabled
 */
export const isReasoningEnabled = (): boolean => {
  return featureFlags.enableReasoning;
};

/**
 * Check if reasoning export is enabled
 */
export const isReasoningExportEnabled = (): boolean => {
  return featureFlags.enableReasoningExport && featureFlags.enableReasoning;
};

/**
 * Check if reasoning comparison is enabled
 */
export const isReasoningComparisonEnabled = (): boolean => {
  return featureFlags.enableReasoningComparison && featureFlags.enableReasoning;
};

/**
 * Get default reasoning level
 */
export const getDefaultReasoningLevel = (): 'basic' | 'detailed' | 'full' => {
  return featureFlags.defaultReasoningLevel;
};

/**
 * Get max thinking steps to display
 */
export const getMaxThinkingSteps = (): number => {
  return featureFlags.maxThinkingSteps;
};

/**
 * Get reasoning cache TTL in milliseconds
 */
export const getReasoningCacheTTL = (): number => {
  return featureFlags.reasoningCacheTTL;
};
