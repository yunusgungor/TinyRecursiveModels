/**
 * Unit tests for useRecommendations hook
 * Tests loading states, error handling, and request cancellation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useRecommendations } from '../useRecommendations';
import { recommendationAPIClient } from '@/lib/api/recommendations';
import type { EnhancedRecommendationResponse, UserProfile } from '@/types/reasoning';

// Mock the API client
vi.mock('@/lib/api/recommendations', () => ({
  recommendationAPIClient: {
    fetchRecommendations: vi.fn(),
  },
}));

describe('useRecommendations', () => {
  const mockUserProfile: UserProfile = {
    hobbies: ['cooking', 'reading'],
    age: 30,
    budget: 500,
    occasion: 'birthday',
  };

  const mockResponse: EnhancedRecommendationResponse = {
    recommendations: [
      {
        gift: {
          id: '1',
          name: 'Test Gift',
          price: 100,
          category: 'Books',
        },
        reasoning: ['Perfect for book lovers'],
        confidence: 0.85,
      },
    ],
    tool_results: {},
    inference_time: 0.5,
    cache_hit: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with empty state', () => {
    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    expect(result.current.recommendations).toEqual([]);
    expect(result.current.toolResults).toEqual({});
    expect(result.current.reasoningTrace).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should handle loading state correctly', async () => {
    // Mock a delayed response
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(mockResponse), 100);
        })
    );

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    // Start fetching
    result.current.refetch();

    // Should be loading
    await waitFor(() => {
      expect(result.current.isLoading).toBe(true);
    });

    // Wait for completion
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.recommendations).toEqual(mockResponse.recommendations);
    expect(result.current.error).toBeNull();
  });

  it('should handle successful fetch', async () => {
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockResolvedValue(mockResponse);

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    await result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.recommendations).toEqual(mockResponse.recommendations);
    expect(result.current.toolResults).toEqual(mockResponse.tool_results);
    expect(result.current.error).toBeNull();
  });

  it('should handle error state correctly', async () => {
    const mockError = new Error('API Error');
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockRejectedValue(mockError);

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    await result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toEqual(mockError);
    expect(result.current.recommendations).toEqual([]);
  });

  it('should handle request cancellation', async () => {
    // Mock a long-running request
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockImplementation(
      ({ signal }) =>
        new Promise((resolve, reject) => {
          const timeout = setTimeout(() => resolve(mockResponse), 1000);
          signal?.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Request cancelled', 'AbortError'));
          });
        })
    );

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    // Start fetching
    result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(true);
    });

    // Cancel the request
    result.current.cancel();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Should not have set error or data
    expect(result.current.error).toBeNull();
    expect(result.current.recommendations).toEqual([]);
  });

  it('should pass correct options to API client', async () => {
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockResolvedValue(mockResponse);

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
        includeReasoning: true,
        reasoningLevel: 'full',
        maxRecommendations: 10,
      })
    );

    await result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(recommendationAPIClient.fetchRecommendations).toHaveBeenCalledWith(
      mockUserProfile,
      expect.objectContaining({
        includeReasoning: true,
        reasoningLevel: 'full',
        maxRecommendations: 10,
      })
    );
  });

  it('should use default options when not provided', async () => {
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockResolvedValue(mockResponse);

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    await result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(recommendationAPIClient.fetchRecommendations).toHaveBeenCalledWith(
      mockUserProfile,
      expect.objectContaining({
        includeReasoning: true,
        reasoningLevel: 'detailed',
        maxRecommendations: 5,
      })
    );
  });

  it('should cancel previous request when refetch is called', async () => {
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(mockResponse), 500);
        })
    );

    const { result } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    // Start first request
    result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(true);
    });

    // Start second request (should cancel first)
    result.current.refetch();

    // Should still be loading (second request)
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Should have completed successfully
    expect(result.current.recommendations).toEqual(mockResponse.recommendations);
  });

  it('should cleanup on unmount', async () => {
    vi.mocked(recommendationAPIClient.fetchRecommendations).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(mockResponse), 1000);
        })
    );

    const { result, unmount } = renderHook(() =>
      useRecommendations({
        userProfile: mockUserProfile,
      })
    );

    // Start fetching
    result.current.refetch();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(true);
    });

    // Unmount while loading - should not throw errors
    expect(() => unmount()).not.toThrow();
  });
});
