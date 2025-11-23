import { useQuery } from '@tanstack/react-query';
import { recommendationsApi } from '@/lib/api/recommendations';

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => recommendationsApi.getHealth(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}
