/**
 * Custom hook for fetching gift recommendations with reasoning
 * Supports reasoning levels, loading states, and request cancellation
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { recommendationAPIClient } from '@/lib/api/recommendations';
import type { 
  EnhancedGiftRecommendation, 
  ReasoningTrace, 
  UserProfile,
  ReasoningLevel 
} from '@/types/reasoning';

export interface UseRecommendationsOptions {
  userProfile: UserProfile;
  includeReasoning?: boolean;
  reasoningLevel?: ReasoningLevel;
  maxRecommendations?: number;
}

export interface UseRecommendationsReturn {
  recommendations: EnhancedGiftRecommendation[];
  toolResults: Record<string, any>;
  reasoningTrace: ReasoningTrace | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  cancel: () => void;
}

/**
 * Hook for fetching recommendations with reasoning
 * 
 * @param options - Configuration options including user profile and reasoning level
 * @returns Recommendations data, loading state, and control functions
 * 
 * @example
 * ```tsx
 * const { recommendations, isLoading, error, refetch } = useRecommendations({
 *   userProfile: { hobbies: ['cooking'], budget: 500 },
 *   reasoningLevel: 'detailed',
 * });
 * 
 * if (isLoading) return <Spinner />;
 * if (error) return <ErrorMessage error={error} onRetry={refetch} />;
 * 
 * return (
 *   <div>
 *     {recommendations.map(rec => (
 *       <GiftCard key={rec.gift.id} recommendation={rec} />
 *     ))}
 *   </div>
 * );
 * ```
 */
export function useRecommendations(
  options: UseRecommendationsOptions
): UseRecommendationsReturn {
  const {
    userProfile,
    includeReasoning = true,
    reasoningLevel = 'detailed',
    maxRecommendations = 5,
  } = options;

  const [recommendations, setRecommendations] = useState<EnhancedGiftRecommendation[]>([]);
  const [toolResults, setToolResults] = useState<Record<string, any>>({});
  const [reasoningTrace, setReasoningTrace] = useState<ReasoningTrace | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // AbortController for request cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch recommendations
  const fetchRecommendations = useCallback(async () => {
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setIsLoading(true);
    setError(null);

    try {
      const response = await recommendationAPIClient.fetchRecommendations(
        userProfile,
        {
          includeReasoning,
          reasoningLevel,
          maxRecommendations,
          signal: abortController.signal,
        }
      );

      // Only update state if request wasn't cancelled
      if (!abortController.signal.aborted) {
        setRecommendations(response.recommendations);
        setToolResults(response.tool_results);
        setReasoningTrace(response.reasoning_trace || null);
        setIsLoading(false);
      }
    } catch (err) {
      // Don't set error if request was cancelled
      if (!abortController.signal.aborted) {
        setError(err instanceof Error ? err : new Error('Failed to fetch recommendations'));
        setIsLoading(false);
      }
    }
  }, [userProfile, includeReasoning, reasoningLevel, maxRecommendations]);

  // Cancel ongoing request
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    recommendations,
    toolResults,
    reasoningTrace,
    isLoading,
    error,
    refetch: fetchRecommendations,
    cancel,
  };
}
