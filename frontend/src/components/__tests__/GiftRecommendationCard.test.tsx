import { describe, test, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GiftRecommendationCard } from '../GiftRecommendationCard';
import { EnhancedGiftRecommendation } from '@/types/reasoning';

/**
 * Unit tests for GiftRecommendationCard component
 */

const mockRecommendation: EnhancedGiftRecommendation = {
  gift: {
    id: '1',
    name: 'Premium Kahve Makinesi',
    price: 2499.99,
    image_url: 'https://example.com/image.jpg',
    category: 'Ev & Yaşam',
    rating: 4.5,
    availability: true,
  },
  reasoning: [
    'Kullanıcının hobi listesinde kahve yapımı var',
    'Bütçe aralığına uygun fiyat',
    'Yaş grubuna uygun hediye',
  ],
  confidence: 0.85,
};

const mockToolResults = {
  review_analysis: {
    average_rating: 4.5,
    review_count: 1234,
  },
  trend_analysis: {
    trending: true,
    trend_score: 0.8,
  },
  inventory_check: {
    available: true,
    stock_count: 50,
  },
};

describe('GiftRecommendationCard', () => {
  describe('Rendering with various gift data', () => {
    test('renders gift name correctly', () => {
      render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      expect(screen.getByText('Premium Kahve Makinesi')).toBeTruthy();
    });

    test('renders gift price in Turkish Lira format', () => {
      const { container } = render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      // Check for Turkish Lira formatting
      expect(container.textContent).toMatch(/2\.499,99/);
      expect(container.textContent).toMatch(/₺/);
    });

    test('renders gift category badge', () => {
      render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      expect(screen.getByText('Ev & Yaşam')).toBeTruthy();
    });

    test('renders confidence indicator', () => {
      const { container } = render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      const confidenceIndicator = container.querySelector('[aria-label*="Güven skoru"]');
      expect(confidenceIndicator).toBeTruthy();
      expect(container.textContent).toContain('85%');
    });

    test('renders reasoning strings', () => {
      const { container } = render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      // Text is split by highlighting, so check container text content
      expect(container.textContent).toContain('hobi listesinde kahve yapımı var');
      expect(container.textContent).toContain('Bütçe aralığına uygun fiyat');
    });

    test('renders tool insights when provided', () => {
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          toolResults={mockToolResults}
        />
      );
      
      // Check for rating
      expect(container.querySelector('[aria-label*="Rating"]')).toBeTruthy();
      expect(container.textContent).toContain('4.5');
      
      // Check for trending
      expect(container.querySelector('[aria-label="Trending"]')).toBeTruthy();
      expect(container.textContent).toContain('Trend');
      
      // Check for availability
      expect(container.querySelector('[aria-label="In Stock"]')).toBeTruthy();
      expect(container.textContent).toContain('Stokta');
    });

    test('does not render tool insights when not provided', () => {
      const { container } = render(
        <GiftRecommendationCard recommendation={mockRecommendation} />
      );
      
      expect(container.querySelector('[aria-label*="Rating"]')).toBeFalsy();
      expect(container.querySelector('[aria-label="Trending"]')).toBeFalsy();
      expect(container.querySelector('[aria-label="In Stock"]')).toBeFalsy();
    });

    test('renders with partial tool results', () => {
      const partialToolResults = {
        review_analysis: {
          average_rating: 4.2,
          review_count: 500,
        },
      };
      
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          toolResults={partialToolResults}
        />
      );
      
      expect(container.querySelector('[aria-label*="Rating"]')).toBeTruthy();
      expect(container.textContent).toContain('4.2');
      expect(container.querySelector('[aria-label="Trending"]')).toBeFalsy();
    });

    test('applies selected styling when isSelected is true', () => {
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          isSelected={true}
        />
      );
      
      const card = container.querySelector('[role="article"]');
      expect(card?.className).toContain('ring-2');
      expect(card?.className).toContain('ring-blue-500');
    });

    test('does not apply selected styling when isSelected is false', () => {
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          isSelected={false}
        />
      );
      
      const card = container.querySelector('[role="article"]');
      expect(card?.className).not.toContain('ring-2');
    });
  });

  describe('Expand/collapse functionality', () => {
    test('shows expand button when reasoning text is long', () => {
      const longRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: [
          'Bu çok uzun bir reasoning metnidir ve kullanıcının hobi listesinde kahve yapımı var',
          'Bütçe aralığına uygun fiyat ve kullanıcının belirlediği maksimum bütçenin altında',
          'Yaş grubuna uygun sofistike bir hediye, 30-40 yaş arası kullanıcılar için ideal',
          'Yüksek kaliteli malzeme ve dayanıklılık özellikleri',
        ],
      };
      
      render(<GiftRecommendationCard recommendation={longRecommendation} />);
      
      expect(screen.getByText('Daha fazla göster')).toBeTruthy();
    });

    test('does not show expand button when reasoning text is short', () => {
      const shortRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: ['Kısa reasoning'],
      };
      
      const { container } = render(<GiftRecommendationCard recommendation={shortRecommendation} />);
      
      expect(container.querySelector('[aria-expanded]')).toBeFalsy();
    });

    test('expands reasoning when expand button is clicked', async () => {
      const user = userEvent.setup();
      const longRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: [
          'Reasoning 1 - Bu çok uzun bir reasoning metnidir ve kullanıcının hobi listesinde kahve yapımı var',
          'Reasoning 2 - Bütçe aralığına uygun fiyat ve kullanıcının belirlediği maksimum bütçenin altında',
          'Reasoning 3 - Yaş grubuna uygun hediye ve sofistike bir seçim',
          'Reasoning 4 - Yüksek kaliteli malzeme ve dayanıklılık özellikleri',
        ],
      };
      
      const { container } = render(<GiftRecommendationCard recommendation={longRecommendation} />);
      
      const expandButton = screen.getByText('Daha fazla göster');
      expect(expandButton.getAttribute('aria-expanded')).toBe('false');
      
      await user.click(expandButton);
      
      // After expansion, button text should change
      expect(screen.getByText('Daha az göster')).toBeTruthy();
      expect(screen.getByText('Daha az göster').getAttribute('aria-expanded')).toBe('true');
      
      // All reasoning items should be visible
      expect(container.textContent).toContain('Reasoning 3');
      expect(container.textContent).toContain('Reasoning 4');
    });

    test('collapses reasoning when collapse button is clicked', async () => {
      const user = userEvent.setup();
      const longRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: [
          'Reasoning 1 - Bu çok uzun bir reasoning metnidir ve kullanıcının hobi listesinde kahve yapımı var',
          'Reasoning 2 - Bütçe aralığına uygun fiyat ve kullanıcının belirlediği maksimum bütçenin altında',
          'Reasoning 3 - Yaş grubuna uygun hediye ve sofistike bir seçim',
          'Reasoning 4 - Yüksek kaliteli malzeme ve dayanıklılık özellikleri',
        ],
      };
      
      render(<GiftRecommendationCard recommendation={longRecommendation} />);
      
      // Expand first
      const expandButton = screen.getByText('Daha fazla göster');
      await user.click(expandButton);
      
      // Then collapse
      const collapseButton = screen.getByText('Daha az göster');
      await user.click(collapseButton);
      
      // Button text should change back
      expect(screen.getByText('Daha fazla göster')).toBeTruthy();
    });
  });

  describe('Button click handlers', () => {
    test('calls onShowDetails when "Detaylı Analiz Göster" button is clicked', async () => {
      const user = userEvent.setup();
      const handleShowDetails = vi.fn();
      
      render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          onShowDetails={handleShowDetails}
        />
      );
      
      const detailsButton = screen.getByText('Detaylı Analiz Göster');
      await user.click(detailsButton);
      
      expect(handleShowDetails).toHaveBeenCalledTimes(1);
    });

    test('does not render "Detaylı Analiz Göster" button when onShowDetails is not provided', () => {
      render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      expect(screen.queryByText('Detaylı Analiz Göster')).toBeFalsy();
    });

    test('calls onSelect when selection checkbox is toggled', async () => {
      const user = userEvent.setup();
      const handleSelect = vi.fn();
      
      render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          onSelect={handleSelect}
          isSelected={false}
        />
      );
      
      const checkbox = screen.getByRole('checkbox', { name: /Karşılaştırma için seç/ });
      await user.click(checkbox);
      
      expect(handleSelect).toHaveBeenCalledTimes(1);
    });

    test('does not render selection checkbox when onSelect is not provided', () => {
      render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      expect(screen.queryByRole('checkbox')).toBeFalsy();
    });

    test('checkbox reflects isSelected state', () => {
      const { rerender } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          onSelect={() => {}}
          isSelected={false}
        />
      );
      
      let checkbox = screen.getByRole('checkbox') as HTMLInputElement;
      expect(checkbox.checked).toBe(false);
      
      rerender(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          onSelect={() => {}}
          isSelected={true}
        />
      );
      
      checkbox = screen.getByRole('checkbox') as HTMLInputElement;
      expect(checkbox.checked).toBe(true);
    });
  });

  describe('Reasoning factor highlighting', () => {
    test('highlights hobby-related keywords', () => {
      const hobbyRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: ['Kullanıcının hobi listesinde bu ürün var'],
      };
      
      const { container } = render(<GiftRecommendationCard recommendation={hobbyRecommendation} />);
      
      const purpleElements = container.querySelectorAll('.text-purple-600, .dark\\:text-purple-400');
      expect(purpleElements.length).toBeGreaterThan(0);
    });

    test('highlights budget-related keywords', () => {
      const budgetRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: ['Bütçe aralığına uygun fiyat'],
      };
      
      const { container } = render(<GiftRecommendationCard recommendation={budgetRecommendation} />);
      
      const greenElements = container.querySelectorAll('.text-green-600, .dark\\:text-green-400');
      expect(greenElements.length).toBeGreaterThan(0);
    });

    test('highlights age-related keywords', () => {
      const ageRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: ['Yaş grubuna uygun hediye'],
      };
      
      const { container } = render(<GiftRecommendationCard recommendation={ageRecommendation} />);
      
      const blueElements = container.querySelectorAll('.text-blue-600, .dark\\:text-blue-400');
      expect(blueElements.length).toBeGreaterThan(0);
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', () => {
      const { container } = render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      const card = container.querySelector('[role="article"]');
      expect(card?.getAttribute('aria-label')).toContain('Gift recommendation');
      expect(card?.getAttribute('aria-label')).toContain('Premium Kahve Makinesi');
    });

    test('reasoning section has proper ARIA label', () => {
      const { container } = render(<GiftRecommendationCard recommendation={mockRecommendation} />);
      
      const reasoningSection = container.querySelector('[aria-label="Reasoning information"]');
      expect(reasoningSection).toBeTruthy();
    });

    test('expand button has proper ARIA attributes', () => {
      const longRecommendation: EnhancedGiftRecommendation = {
        ...mockRecommendation,
        reasoning: [
          'Bu çok uzun bir reasoning metnidir ve kullanıcının hobi listesinde kahve yapımı var',
          'Bütçe aralığına uygun fiyat ve kullanıcının belirlediği maksimum bütçenin altında',
          'Yaş grubuna uygun sofistike bir hediye',
        ],
      };
      
      render(<GiftRecommendationCard recommendation={longRecommendation} />);
      
      const expandButton = screen.getByText('Daha fazla göster');
      expect(expandButton.getAttribute('aria-expanded')).toBe('false');
      expect(expandButton.getAttribute('aria-label')).toBe('Daha fazla göster');
    });

    test('tool insights have proper ARIA labels', () => {
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          toolResults={mockToolResults}
        />
      );
      
      const toolInsightsList = container.querySelector('[aria-label="Tool insights"]');
      expect(toolInsightsList).toBeTruthy();
      expect(toolInsightsList?.getAttribute('role')).toBe('list');
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      expect(listItems.length).toBeGreaterThan(0);
    });
  });

  describe('Custom className', () => {
    test('applies custom className to card', () => {
      const { container } = render(
        <GiftRecommendationCard
          recommendation={mockRecommendation}
          className="custom-test-class"
        />
      );
      
      const card = container.querySelector('[role="article"]');
      expect(card?.className).toContain('custom-test-class');
    });
  });
});
