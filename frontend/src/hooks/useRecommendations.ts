import { useMutation } from '@tanstack/react-query';
import { recommendationsApi } from '@/lib/api/recommendations';
import type { RecommendationRequest } from '@/lib/api/types';

export function useRecommendations() {
  return useMutation({
    mutationFn: (request: RecommendationRequest) =>
      recommendationsApi.getRecommendations(request),
  });
}
