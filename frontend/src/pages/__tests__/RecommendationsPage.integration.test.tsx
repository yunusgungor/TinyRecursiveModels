/**
 * Integration tests for RecommendationsPage
 * 
 * Tests:
 * - Full user flow from loading to displaying recommendations
 * - Reasoning panel interaction
 * - Comparison mode functionality
 * - Loading and error states
 * - Gift selection and deselection
 * 
 * Requirements: 1.1, 7.1, 7.2, 12.1, 12.2, 12.5
 */

import { describe, test, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReasoningProvider } from '@/contexts/ReasoningContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { RecommendationsPage } from '../RecommendationsPage';
import * as useRecommendationsHook from '@/hooks/useRecommendations';
import type {
  UserProfile,
  EnhancedGiftRecommendation,
  ReasoningTrace,
} from '@/types/reasoning';

// Mock the useRecommendations hook
vi.mock('@/hooks/useRecommendations');

// Create a mock store state that can be controlled per test
const mockStoreState = {
  selectedGiftsForComparison: [] as string[],
  isComparisonMode: false,
  toggleGiftSelection: vi.fn((id: string) => {
    const index = mockStoreState.selectedGiftsForComparison.indexOf(id);
    if (index > -1) {
      mockStoreState.selectedGiftsForComparison.splice(index, 1);
    } else {
      mockStoreState.selectedGiftsForComparison.push(id);
    }
  }),
  clearGiftSelection: vi.fn(() => {
    mockStoreState.selectedGiftsForComparison = [];
    mockStoreState.isComparisonMode = false;
  }),
  setComparisonMode: vi.fn((enabled: boolean) => {
    mockStoreState.isComparisonMode = enabled;
  }),
};

// Mock the useAppStore to prevent state sharing between tests
vi.mock('@/store/useAppStore', () => ({
  useAppStore: vi.fn(() => mockStoreState),
}));

const mockUserProfile: UserProfile = {
  age: 25,
  hobbies: ['Spor', 'Müzik'],
  budget: 500,
  occasion: 'Doğum Günü',
  relationship: 'Arkadaş',
  gender: 'Erkek',
};

const mockReasoningTrace: ReasoningTrace = {
  tool_selection: [
    {
      name: 'review_analysis',
      selected: true,
      score: 0.9,
      reason: 'Yüksek güvenilirlik',
      confidence: 0.85,
      priority: 1,
    },
    {
      name: 'trend_analysis',
      selected: false,
      score: 0.4,
      reason: 'Düşük öncelik',
      confidence: 0.45,
      priority: 3,
    },
  ],
  category_matching: [
    {
      category_name: 'Spor Malzemeleri',
      score: 0.85,
      reasons: ['Hobi eşleşmesi', 'Yaş uygunluğu'],
      feature_contributions: {
        hobbies: 0.6,
        age: 0.25,
      },
    },
    {
      category_name: 'Elektronik',
      score: 0.65,
      reasons: ['Bütçe uygunluğu'],
      feature_contributions: {
        budget: 0.5,
      },
    },
  ],
  attention_weights: {
    user_features: {
      hobbies: 0.4,
      budget: 0.3,
      age: 0.2,
      occasion: 0.1,
    },
    gift_features: {
      category: 0.5,
      price: 0.3,
      rating: 0.2,
    },
  },
  thinking_steps: [
    {
      step: 1,
      action: 'Kullanıcı profilini analiz et',
      result: 'Spor ve müzik ilgi alanları tespit edildi',
      insight: 'Aktif yaşam tarzı',
    },
    {
      step: 2,
      action: 'Kategori eşleştirmesi yap',
      result: 'Spor malzemeleri en yüksek skor',
      insight: 'Hobi bazlı öneri',
    },
  ],
};

const mockRecommendations: EnhancedGiftRecommendation[] = [
  {
    gift: {
      id: 'gift-1',
      name: 'Spor Ayakkabısı',
      category: 'Spor Malzemeleri',
      price: 450,
      rating: 4.5,
      image_url: 'https://example.com/shoe.jpg',
      trendyolUrl: 'https://trendyol.com/shoe',
      description: 'Koşu için ideal',
      tags: ['spor', 'ayakkabı'],
      age_suitability: [18, 65],
      occasion_fit: ['Doğum Günü'],
      in_stock: true,
    },
    reasoning: ['Spor hobisine uygun', 'Bütçe dahilinde', 'Yüksek puan'],
    confidence: 0.85,
    reasoning_trace: mockReasoningTrace,
  },
  {
    gift: {
      id: 'gift-2',
      name: 'Bluetooth Kulaklık',
      category: 'Elektronik',
      price: 350,
      rating: 4.2,
      image_url: 'https://example.com/headphone.jpg',
      trendyolUrl: 'https://trendyol.com/headphone',
      description: 'Müzik dinlemek için',
      tags: ['elektronik', 'müzik'],
      age_suitability: [15, 100],
      occasion_fit: ['Doğum Günü'],
      in_stock: true,
    },
    reasoning: ['Müzik hobisine uygun', 'Uygun fiyat'],
    confidence: 0.72,
    reasoning_trace: mockReasoningTrace,
  },
  {
    gift: {
      id: 'gift-3',
      name: 'Spor Çantası',
      category: 'Spor Malzemeleri',
      price: 200,
      rating: 4.0,
      image_url: 'https://example.com/bag.jpg',
      trendyolUrl: 'https://trendyol.com/bag',
      description: 'Spor salonu için',
      tags: ['spor', 'çanta'],
      age_suitability: [18, 65],
      occasion_fit: ['Doğum Günü'],
      in_stock: true,
    },
    reasoning: ['Spor aktiviteleri için pratik'],
    confidence: 0.68,
    reasoning_trace: mockReasoningTrace,
  },
];

const mockToolResults = {
  review_analysis: {
    average_rating: 4.5,
    review_count: 1250,
  },
  trend_analysis: {
    trending: true,
    trend_score: 0.85,
  },
};

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <ThemeProvider>
      <ReasoningProvider>{component}</ReasoningProvider>
    </ThemeProvider>
  );
};

describe('RecommendationsPage Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Clear localStorage to reset reasoning context state
    localStorage.clear();
    // Reset mock store state
    mockStoreState.selectedGiftsForComparison = [];
    mockStoreState.isComparisonMode = false;
  });

  describe('Full User Flow', () => {
    test('successfully loads and displays recommendations', async () => {
      // Mock successful recommendations fetch
      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Check page title
      expect(screen.getByText('Hediye Önerileri')).toBeInTheDocument();

      // Check recommendations count
      expect(screen.getByText('3 öneri bulundu')).toBeInTheDocument();

      // Check all recommendations are displayed
      expect(screen.getByText('Spor Ayakkabısı')).toBeInTheDocument();
      expect(screen.getByText('Bluetooth Kulaklık')).toBeInTheDocument();
      expect(screen.getByText('Spor Çantası')).toBeInTheDocument();

      // Check reasoning is displayed
      const reasoningRegions = screen.getAllByRole('region', { name: /reasoning information/i });
      expect(reasoningRegions.length).toBeGreaterThan(0);
    });

    test('displays loading state while fetching', () => {
      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: [],
        toolResults: {},
        reasoningTrace: null,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Check loading state is displayed
      expect(screen.getByText(/öneriler yükleniyor/i)).toBeInTheDocument();
      expect(screen.getAllByRole('status', { name: /loading gift recommendation/i })).toHaveLength(3);
    });

    test('displays error state when fetch fails', () => {
      const mockError = new Error('API hatası');
      const mockRefetch = vi.fn();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: [],
        toolResults: {},
        reasoningTrace: null,
        isLoading: false,
        error: mockError,
        refetch: mockRefetch,
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Check error message is displayed
      expect(screen.getByText(/öneriler yüklenemedi/i)).toBeInTheDocument();
      expect(screen.getByText(/API hatası/i)).toBeInTheDocument();

      // Check retry button exists
      const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
      expect(retryButton).toBeInTheDocument();
    });

    test('displays empty state when no recommendations found', () => {
      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: [],
        toolResults: {},
        reasoningTrace: null,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Check empty state message
      expect(screen.getByText(/öneri bulunamadı/i)).toBeInTheDocument();
      expect(
        screen.getByText(/belirttiğiniz kriterlere uygun hediye bulunamadı/i)
      ).toBeInTheDocument();
    });
  });

  describe('Reasoning Panel Interaction', () => {
    test('opens reasoning panel when clicking show details button', async () => {
      const user = userEvent.setup();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Find and click the first "Detaylı Analiz Göster" button
      const detailsButtons = screen.getAllByRole('button', {
        name: /detaylı analiz göster/i,
      });
      await user.click(detailsButtons[0]);

      // Check that reasoning panel opens
      await waitFor(() => {
        expect(screen.getByText('Detaylı Analiz')).toBeInTheDocument();
      });

      // Check that reasoning panel title is visible
      expect(screen.getByText('Detaylı Analiz')).toBeInTheDocument();
      
      // Check that filter buttons are visible
      const filterButtons = screen.getAllByRole('button');
      const filterTexts = filterButtons.map(btn => btn.textContent);
      expect(filterTexts.some(text => text?.includes('Tool Seçimi'))).toBe(true);
    });

    test('closes reasoning panel when clicking close button', async () => {
      const user = userEvent.setup();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Open panel
      const detailsButtons = screen.getAllByRole('button', {
        name: /detaylı analiz göster/i,
      });
      await user.click(detailsButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Detaylı Analiz')).toBeInTheDocument();
      });

      // Close panel
      const closeButton = screen.getByRole('button', { name: /paneli kapat/i });
      await user.click(closeButton);

      // Panel should be closed
      await waitFor(() => {
        expect(screen.queryByText('Detaylı Analiz')).not.toBeInTheDocument();
      });
    });

    test('displays all reasoning sections in panel', async () => {
      const user = userEvent.setup();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Open panel
      const detailsButtons = screen.getAllByRole('button', {
        name: /detaylı analiz göster/i,
      });
      await user.click(detailsButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Detaylı Analiz')).toBeInTheDocument();
      });

      // Check all filter sections are present
      const filterButtons = screen.getAllByRole('button');
      const filterTexts = filterButtons.map(btn => btn.textContent);
      expect(filterTexts.some(text => text?.includes('Tool Seçimi'))).toBe(true);
      expect(filterTexts.some(text => text?.includes('Kategori Eşleştirme'))).toBe(true);
      expect(filterTexts.some(text => text?.includes('Attention Weights'))).toBe(true);
      expect(filterTexts.some(text => text?.includes('Düşünme Adımları'))).toBe(true);
    });
  });

  describe('Comparison Mode', () => {
    test('renders gift selection checkboxes', async () => {
      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(<RecommendationsPage userProfile={mockUserProfile} />);

      // Wait for recommendations to load
      await waitFor(() => {
        expect(screen.getByText('Spor Ayakkabısı')).toBeInTheDocument();
      });

      // Check that selection checkboxes are rendered
      const checkboxes = screen.getAllByRole('checkbox', {
        name: /karşılaştırma için seç/i,
      });
      expect(checkboxes).toHaveLength(3);
    });
  });

  describe('Back to Search', () => {
    test('calls onBackToSearch when clicking back button', async () => {
      const user = userEvent.setup();
      const mockOnBackToSearch = vi.fn();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: mockRecommendations,
        toolResults: mockToolResults,
        reasoningTrace: mockReasoningTrace,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(
        <RecommendationsPage
          userProfile={mockUserProfile}
          onBackToSearch={mockOnBackToSearch}
        />
      );

      // Wait for recommendations to load
      await waitFor(() => {
        expect(screen.getByText('Spor Ayakkabısı')).toBeInTheDocument();
      });

      // Click back button (aria-label is "Aramaya geri dön")
      const backButton = screen.getByRole('button', { name: /aramaya geri dön/i });
      await user.click(backButton);

      // Check callback was called
      expect(mockOnBackToSearch).toHaveBeenCalledTimes(1);
    });

    test('shows back button in empty state', () => {
      const mockOnBackToSearch = vi.fn();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: [],
        toolResults: {},
        reasoningTrace: null,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
        cancel: vi.fn(),
      });

      renderWithProviders(
        <RecommendationsPage
          userProfile={mockUserProfile}
          onBackToSearch={mockOnBackToSearch}
        />
      );

      // Check back button exists in empty state
      expect(screen.getByRole('button', { name: /yeni arama yap/i })).toBeInTheDocument();
    });
  });

  describe('Request Cancellation', () => {
    test('cancels request on unmount', () => {
      const mockCancel = vi.fn();

      vi.mocked(useRecommendationsHook.useRecommendations).mockReturnValue({
        recommendations: [],
        toolResults: {},
        reasoningTrace: null,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
        cancel: mockCancel,
      });

      const { unmount } = renderWithProviders(
        <RecommendationsPage userProfile={mockUserProfile} />
      );

      // Unmount component
      unmount();

      // Check that cancel was called
      expect(mockCancel).toHaveBeenCalled();
    });
  });
});
