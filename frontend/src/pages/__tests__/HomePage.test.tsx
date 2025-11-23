import { describe, test, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { HomePage } from '../HomePage';
import * as recommendationsApi from '@/lib/api/recommendations';
import type { RecommendationResponse } from '@/lib/api/types';

// Mock the recommendations API
vi.mock('@/lib/api/recommendations', () => ({
  recommendationsApi: {
    getRecommendations: vi.fn(),
  },
}));

// Mock window.open
const mockWindowOpen = vi.fn();
window.open = mockWindowOpen;

// Mock window.scrollTo
window.scrollTo = vi.fn();

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{component}</ThemeProvider>
    </QueryClientProvider>
  );
};

const mockRecommendationResponse: RecommendationResponse = {
  recommendations: [
    {
      gift: {
        id: '1',
        name: 'Test Ürün 1',
        category: 'Elektronik',
        price: 500,
        rating: 4.5,
        imageUrl: 'https://example.com/image1.jpg',
        trendyolUrl: 'https://trendyol.com/product1',
        description: 'Test açıklama',
        tags: ['test'],
        ageSuitability: [18, 65],
        occasionFit: ['Doğum Günü'],
        inStock: true,
      },
      confidenceScore: 0.85,
      reasoning: ['Kullanıcı profiline uygun', 'Bütçe dahilinde'],
      toolInsights: {},
      rank: 1,
    },
    {
      gift: {
        id: '2',
        name: 'Test Ürün 2',
        category: 'Kitap',
        price: 150,
        rating: 4.2,
        imageUrl: 'https://example.com/image2.jpg',
        trendyolUrl: 'https://trendyol.com/product2',
        description: 'Test açıklama 2',
        tags: ['test'],
        ageSuitability: [18, 100],
        occasionFit: ['Doğum Günü'],
        inStock: true,
      },
      confidenceScore: 0.45,
      reasoning: ['Alternatif seçenek'],
      toolInsights: {},
      rank: 2,
    },
  ],
  toolResults: {
    priceComparison: {
      bestPrice: 450,
      averagePrice: 500,
      priceRange: '450-550 TL',
      savingsPercentage: 10,
      checkedPlatforms: ['Trendyol', 'Hepsiburada'],
    },
  },
  inferenceTime: 1.5,
  cacheHit: false,
};

describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Page Rendering', () => {
    test('renders page title and description', () => {
      renderWithQueryClient(<HomePage />);

      expect(screen.getByText(/trendyol hediye önerisi/i)).toBeInTheDocument();
      expect(
        screen.getByText(/sevdikleriniz için mükemmel hediyeyi bulun/i)
      ).toBeInTheDocument();
    });

    test('renders user profile form initially', () => {
      renderWithQueryClient(<HomePage />);

      expect(screen.getByLabelText(/yaş/i)).toBeInTheDocument();
      expect(screen.getByText(/hobiler/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /hediye önerisi al/i })).toBeInTheDocument();
    });

    test('does not render recommendations initially', () => {
      renderWithQueryClient(<HomePage />);

      expect(screen.queryByText(/sizin için seçtiklerimiz/i)).not.toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    test('shows loading state when fetching recommendations', async () => {
      const user = userEvent.setup();
      
      // Mock API to delay response
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );

      renderWithQueryClient(<HomePage />);

      // Fill and submit form
      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      // Check loading state
      await waitFor(() => {
        expect(screen.getByText(/öneriler hazırlanıyor/i)).toBeInTheDocument();
      });

      expect(
        screen.getByText(/yapay zeka modelimiz sizin için en uygun hediye önerilerini araştırıyor/i)
      ).toBeInTheDocument();
    });

    test('hides form during loading', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockImplementation(
        () => new Promise(() => {})
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.queryByLabelText(/yaş/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Flow Integration', () => {
    test('successfully fetches and displays recommendations', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue(
        mockRecommendationResponse
      );

      renderWithQueryClient(<HomePage />);

      // Fill and submit form
      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      // Wait for recommendations to appear
      await waitFor(() => {
        expect(screen.getByText('Test Ürün 1')).toBeInTheDocument();
      });

      expect(screen.getByText('Test Ürün 2')).toBeInTheDocument();
      expect(screen.getByText(/öneri bulundu/i)).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'Sizin İçin Seçtiklerimiz' })).toBeInTheDocument();
    });

    test('displays inference time and cache status', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue(
        mockRecommendationResponse
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText('Test Ürün 1')).toBeInTheDocument();
      });

      // Check that inference time is displayed (using getAllByText and checking one contains the full text)
      const elements = screen.getAllByText((content, element) => {
        return element?.textContent?.includes('İşlem süresi: 1.50s') || false;
      });
      expect(elements.length).toBeGreaterThan(0);
    });

    test('opens details modal when clicking details button', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue(
        mockRecommendationResponse
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText('Test Ürün 1')).toBeInTheDocument();
      });

      const detailsButtons = screen.getAllByRole('button', { name: /detaylar/i });
      await user.click(detailsButtons[0]);

      // Modal should open with product details
      await waitFor(() => {
        expect(screen.getByText(/öneri gerekçesi/i)).toBeInTheDocument();
      });
    });

    test('opens Trendyol link when clicking Trendyol button', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue(
        mockRecommendationResponse
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText('Test Ürün 1')).toBeInTheDocument();
      });

      const trendyolButtons = screen.getAllByRole('button', { name: /trendyol'da gör/i });
      await user.click(trendyolButtons[0]);

      expect(mockWindowOpen).toHaveBeenCalledWith(
        'https://trendyol.com/product1',
        '_blank',
        'noopener,noreferrer'
      );
    });

    test('allows starting new search after results', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue(
        mockRecommendationResponse
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText('Test Ürün 1')).toBeInTheDocument();
      });

      const newSearchButton = screen.getByRole('button', { name: /yeni arama/i });
      await user.click(newSearchButton);

      expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' });
    });
  });

  describe('Error Handling', () => {
    test('displays error message when API fails', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockRejectedValue({
        response: {
          data: {
            message: 'Model şu anda kullanılamıyor',
            errorCode: 'MODEL_ERROR',
          },
        },
      });

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText(/öneri alınamadı/i)).toBeInTheDocument();
      });

      expect(screen.getByText(/model şu anda kullanılamıyor/i)).toBeInTheDocument();
      expect(screen.getByText(/hata kodu: MODEL_ERROR/i)).toBeInTheDocument();
    });

    test('displays generic error message for unknown errors', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockRejectedValue(
        new Error('Network error')
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText(/öneri alınamadı/i)).toBeInTheDocument();
      });

      expect(screen.getByText(/network error/i)).toBeInTheDocument();
    });

    test('shows reload button on error', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockRejectedValue(
        new Error('API Error')
      );

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /sayfayı yenile/i })).toBeInTheDocument();
      });
    });

    test('displays warning when no recommendations found', async () => {
      const user = userEvent.setup();
      
      vi.mocked(recommendationsApi.recommendationsApi.getRecommendations).mockResolvedValue({
        recommendations: [],
        toolResults: {},
        inferenceTime: 1.0,
        cacheHit: false,
      });

      renderWithQueryClient(<HomePage />);

      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');
      await user.click(screen.getByRole('button', { name: /hediye önerisi al/i }));

      await waitFor(() => {
        expect(screen.getByText(/öneri bulunamadı/i)).toBeInTheDocument();
      });

      expect(
        screen.getByText(/belirttiğiniz kriterlere uygun hediye bulunamadı/i)
      ).toBeInTheDocument();
    });
  });
});
