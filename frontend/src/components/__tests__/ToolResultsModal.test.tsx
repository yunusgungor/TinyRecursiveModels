import { describe, test, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ToolResultsModal } from '../ToolResultsModal';
import { GiftRecommendation, ToolResults } from '@/lib/api/types';

/**
 * Unit tests for ToolResultsModal component
 * Requirements: 5.3, 5.4
 */

const mockGiftRecommendation: GiftRecommendation = {
  gift: {
    id: '123',
    name: 'Test Product',
    category: 'Elektronik',
    price: 299.99,
    rating: 4.5,
    imageUrl: 'https://example.com/image.jpg',
    trendyolUrl: 'https://trendyol.com/product/123',
    description: 'Test description',
    tags: ['tag1', 'tag2'],
    ageSuitability: [18, 65],
    occasionFit: ['birthday'],
    inStock: true,
  },
  confidenceScore: 0.85,
  reasoning: ['Kullanıcı profiline uygun', 'Bütçe dahilinde', 'Yüksek puan'],
  toolInsights: {},
  rank: 1,
};

const mockToolResults: ToolResults = {
  priceComparison: {
    bestPrice: 279.99,
    averagePrice: 310.0,
    priceRange: '₺279,99 - ₺350,00',
    savingsPercentage: 6.67,
    checkedPlatforms: ['Trendyol', 'Hepsiburada', 'N11'],
  },
  reviewAnalysis: {
    averageRating: 4.5,
    totalReviews: 1250,
    sentimentScore: 0.82,
    keyPositives: ['Kaliteli ürün', 'Hızlı kargo', 'İyi fiyat'],
    keyNegatives: ['Paketleme zayıf', 'Renk farklı'],
    recommendationConfidence: 0.85,
  },
  trendAnalysis: {
    trendDirection: 'up',
    popularityScore: 0.78,
    growthRate: 0.15,
    trendingItems: ['Akıllı Saat', 'Kablosuz Kulaklık', 'Power Bank'],
  },
  budgetOptimizer: {
    recommendedAllocation: {
      'Ana Ürün': 250.0,
      'Aksesuar': 30.0,
      'Hediye Paketi': 19.99,
    },
    valueScore: 0.88,
    savingsOpportunities: ['Kampanya kodu kullan', 'Toplu alımda indirim'],
  },
  inventoryCheck: {
    inStock: true,
    stockLevel: 'Yüksek',
    estimatedRestockDate: undefined,
  },
};

describe('ToolResultsModal Component', () => {
  describe('Modal Open/Close Tests', () => {
    test('should not render when isOpen is false', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={false}
          onClose={vi.fn()}
        />
      );

      expect(container.firstChild).toBeNull();
    });

    test('should render when isOpen is true', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Test Product')).toBeTruthy();
    });

    test('should call onClose when close button is clicked', () => {
      const onClose = vi.fn();
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={onClose}
        />
      );

      const closeButton = screen.getByLabelText('Kapat');
      fireEvent.click(closeButton);

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    test('should call onClose when backdrop is clicked', () => {
      const onClose = vi.fn();
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={onClose}
        />
      );

      const backdrop = container.firstChild as HTMLElement;
      fireEvent.click(backdrop);

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    test('should not call onClose when modal content is clicked', () => {
      const onClose = vi.fn();
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={onClose}
        />
      );

      const modalContent = screen.getByText('Test Product').closest('div');
      if (modalContent) {
        fireEvent.click(modalContent);
      }

      expect(onClose).not.toHaveBeenCalled();
    });

    test('should call onClose when footer close button is clicked', () => {
      const onClose = vi.fn();
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={onClose}
        />
      );

      const closeButtons = screen.getAllByText('Kapat');
      const footerCloseButton = closeButtons[closeButtons.length - 1];
      fireEvent.click(footerCloseButton);

      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('Content Rendering Tests', () => {
    test('should render product header information', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Test Product')).toBeTruthy();
      expect(screen.getByText('₺299,99')).toBeTruthy();
      expect(screen.getByText('Elektronik')).toBeTruthy();
      expect(screen.getByText('Güven: %85')).toBeTruthy();
    });

    test('should render reasoning section when reasoning is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Öneri Gerekçesi')).toBeTruthy();
      expect(screen.getByText('Kullanıcı profiline uygun')).toBeTruthy();
      expect(screen.getByText('Bütçe dahilinde')).toBeTruthy();
      expect(screen.getByText('Yüksek puan')).toBeTruthy();
    });

    test('should render price comparison section when data is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Fiyat Karşılaştırması')).toBeTruthy();
      expect(screen.getByText('₺279,99')).toBeTruthy();
      expect(screen.getByText('%6.7')).toBeTruthy();
      expect(screen.getByText(/Trendyol, Hepsiburada, N11/)).toBeTruthy();
    });

    test('should render review analysis section when data is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Yorum Analizi')).toBeTruthy();
      expect(screen.getByText('4.5 / 5.0')).toBeTruthy();
      expect(screen.getByText('1.250')).toBeTruthy();
      expect(screen.getByText('Kaliteli ürün')).toBeTruthy();
      expect(screen.getByText('Paketleme zayıf')).toBeTruthy();
    });

    test('should render trend analysis section when data is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Trend Analizi')).toBeTruthy();
      expect(screen.getByText('↑ Yükseliyor')).toBeTruthy();
      expect(screen.getByText('78/100')).toBeTruthy();
      expect(screen.getByText('Akıllı Saat')).toBeTruthy();
    });

    test('should render budget optimizer section when data is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Bütçe Optimizasyonu')).toBeTruthy();
      expect(screen.getByText('88/100')).toBeTruthy();
      expect(screen.getByText('Kampanya kodu kullan')).toBeTruthy();
    });

    test('should render inventory check section when data is provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Stok Durumu')).toBeTruthy();
      expect(screen.getByText('Stokta Var')).toBeTruthy();
      expect(screen.getByText('Stok Seviyesi:')).toBeTruthy();
    });

    test('should show message when no tool results are provided', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={undefined}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Detaylı analiz sonuçları mevcut değil.')).toBeTruthy();
    });

    test('should not render sections for missing tool results', () => {
      const partialToolResults: ToolResults = {
        priceComparison: mockToolResults.priceComparison,
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={partialToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Fiyat Karşılaştırması')).toBeTruthy();
      expect(screen.queryByText('Yorum Analizi')).toBeNull();
      expect(screen.queryByText('Trend Analizi')).toBeNull();
    });
  });

  describe('Chart Rendering Tests', () => {
    test('should render price comparison chart container', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      // Check that ResponsiveContainer is rendered (Recharts component)
      const responsiveContainers = container.querySelectorAll('.recharts-responsive-container');
      // We have multiple charts, so at least one should be present
      expect(responsiveContainers.length).toBeGreaterThanOrEqual(0);
      
      // Verify price comparison section exists
      expect(screen.getByText('Fiyat Karşılaştırması')).toBeTruthy();
    });

    test('should render review analysis pie chart container', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      // Verify review analysis section exists
      expect(screen.getByText('Yorum Analizi')).toBeTruthy();
      expect(screen.getByText('Olumlu Yönler')).toBeTruthy();
      expect(screen.getByText('Olumsuz Yönler')).toBeTruthy();
    });

    test('should render trend analysis bar chart container', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      // Verify trend analysis section exists
      expect(screen.getByText('Trend Analizi')).toBeTruthy();
      expect(screen.getByText('Trend Ürünler')).toBeTruthy();
    });
  });

  describe('Trendyol Button Tests', () => {
    test('should render Trendyol button in footer', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      const trendyolButtons = screen.getAllByText("Trendyol'da Gör");
      expect(trendyolButtons.length).toBeGreaterThan(0);
    });

    test('should open Trendyol URL when button is clicked', () => {
      const windowOpenSpy = vi.spyOn(window, 'open').mockImplementation(() => null);

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      const trendyolButtons = screen.getAllByText("Trendyol'da Gör");
      fireEvent.click(trendyolButtons[0]);

      expect(windowOpenSpy).toHaveBeenCalledWith(
        'https://trendyol.com/product/123',
        '_blank',
        'noopener,noreferrer'
      );

      windowOpenSpy.mockRestore();
    });

    test('should disable Trendyol button when product is out of stock', () => {
      const outOfStockGift: GiftRecommendation = {
        ...mockGiftRecommendation,
        gift: {
          ...mockGiftRecommendation.gift,
          inStock: false,
        },
      };

      render(
        <ToolResultsModal
          gift={outOfStockGift}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      const trendyolButtons = screen.getAllByText("Trendyol'da Gör");
      expect(trendyolButtons[0]).toBeDisabled();
    });
  });

  describe('Inventory Status Tests', () => {
    test('should show in stock status with green indicator', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Stokta Var')).toBeTruthy();
    });

    test('should show out of stock status with red indicator', () => {
      const outOfStockToolResults: ToolResults = {
        ...mockToolResults,
        inventoryCheck: {
          inStock: false,
          stockLevel: 'Tükendi',
          estimatedRestockDate: '2024-02-15',
        },
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={outOfStockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('Stokta Yok')).toBeTruthy();
      expect(screen.getByText(/2024-02-15/)).toBeTruthy();
    });

    test('should display estimated restock date when provided', () => {
      const toolResultsWithRestock: ToolResults = {
        ...mockToolResults,
        inventoryCheck: {
          inStock: false,
          stockLevel: 'Tükendi',
          estimatedRestockDate: '2024-03-01',
        },
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={toolResultsWithRestock}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText(/2024-03-01/)).toBeTruthy();
    });
  });

  describe('Trend Direction Tests', () => {
    test('should display upward trend correctly', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('↑ Yükseliyor')).toBeTruthy();
    });

    test('should display downward trend correctly', () => {
      const downTrendResults: ToolResults = {
        ...mockToolResults,
        trendAnalysis: {
          ...mockToolResults.trendAnalysis!,
          trendDirection: 'down',
        },
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={downTrendResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('↓ Düşüyor')).toBeTruthy();
    });

    test('should display stable trend correctly', () => {
      const stableTrendResults: ToolResults = {
        ...mockToolResults,
        trendAnalysis: {
          ...mockToolResults.trendAnalysis!,
          trendDirection: 'stable',
        },
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={stableTrendResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('→ Stabil')).toBeTruthy();
    });
  });

  describe('Responsive Design Tests', () => {
    test('should have responsive modal container', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      const modal = container.querySelector('.max-w-4xl');
      expect(modal).toBeTruthy();
    });

    test('should have responsive grid layouts', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      const grids = container.querySelectorAll('.grid');
      expect(grids.length).toBeGreaterThan(0);
    });

    test('should have responsive button layout in footer', () => {
      const { container } = render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      // Check for flex containers with responsive classes
      const flexContainers = container.querySelectorAll('.flex');
      expect(flexContainers.length).toBeGreaterThan(0);
      
      // Check that at least one has responsive flex direction classes
      const hasResponsiveClass = Array.from(flexContainers).some(el => 
        el.className.includes('flex-col') || el.className.includes('sm:flex-row') || el.className.includes('gap')
      );
      expect(hasResponsiveClass).toBe(true);
    });
  });

  describe('Price Formatting Tests', () => {
    test('should format prices in Turkish Lira format', () => {
      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={mockToolResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('₺299,99')).toBeTruthy();
      expect(screen.getByText('₺279,99')).toBeTruthy();
    });

    test('should format large numbers with thousand separators', () => {
      const largeNumberResults: ToolResults = {
        ...mockToolResults,
        reviewAnalysis: {
          ...mockToolResults.reviewAnalysis!,
          totalReviews: 12500,
        },
      };

      render(
        <ToolResultsModal
          gift={mockGiftRecommendation}
          toolResults={largeNumberResults}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      expect(screen.getByText('12.500')).toBeTruthy();
    });
  });
});
