import { describe, test, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RecommendationCard } from '../RecommendationCard';
import { GiftRecommendation } from '@/lib/api/types';

/**
 * Unit tests for RecommendationCard component
 * Requirements: 5.1, 5.2, 8.1
 */

const mockRecommendation: GiftRecommendation = {
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
  reasoning: ['Good match'],
  toolInsights: {},
  rank: 1,
};

describe('RecommendationCard Component', () => {
  describe('Rendering Tests', () => {
    test('should render product card with all basic information', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      // Check product name
      expect(screen.getByText('Test Product')).toBeTruthy();

      // Check category
      expect(screen.getByText('Elektronik')).toBeTruthy();

      // Check price
      expect(screen.getByText('₺299,99')).toBeTruthy();

      // Check rating
      expect(screen.getByText('(4.5)')).toBeTruthy();

      // Check confidence score
      expect(screen.getByText('85%')).toBeTruthy();
    });

    test('should render product image with correct attributes', () => {
      const { container } = render(<RecommendationCard recommendation={mockRecommendation} />);

      const image = container.querySelector('img');
      expect(image).toBeTruthy();
      expect(image?.getAttribute('src')).toBe('https://example.com/image.jpg');
      expect(image?.getAttribute('alt')).toBe('Test Product');
      expect(image?.getAttribute('loading')).toBe('lazy');
    });

    test('should render Trendyol button', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      const button = screen.getByText("Trendyol'da Gör");
      expect(button).toBeTruthy();
      expect(button).not.toBeDisabled();
    });

    test('should render details button when onDetailsClick is provided', () => {
      const onDetailsClick = vi.fn();
      render(
        <RecommendationCard
          recommendation={mockRecommendation}
          onDetailsClick={onDetailsClick}
        />
      );

      const button = screen.getByText('Detaylar');
      expect(button).toBeTruthy();
    });

    test('should not render details button when onDetailsClick is not provided', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      const button = screen.queryByText('Detaylar');
      expect(button).toBeNull();
    });

    test('should display low confidence warning when confidence score is below 0.5', () => {
      const lowConfidenceRec: GiftRecommendation = {
        ...mockRecommendation,
        confidenceScore: 0.3,
      };

      render(<RecommendationCard recommendation={lowConfidenceRec} />);

      const warning = screen.getByText(/düşük güven skoruna sahip/i);
      expect(warning).toBeTruthy();
    });

    test('should not display warning when confidence score is 0.5 or above', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      const warning = screen.queryByText(/düşük güven skoruna sahip/i);
      expect(warning).toBeNull();
    });

    test('should display "Stokta Yok" overlay when product is out of stock', () => {
      const outOfStockRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          inStock: false,
        },
      };

      render(<RecommendationCard recommendation={outOfStockRec} />);

      const outOfStockLabel = screen.getByText('Stokta Yok');
      expect(outOfStockLabel).toBeTruthy();
    });

    test('should disable Trendyol button when product is out of stock', () => {
      const outOfStockRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          inStock: false,
        },
      };

      render(<RecommendationCard recommendation={outOfStockRec} />);

      const button = screen.getByText("Trendyol'da Gör");
      expect(button).toBeDisabled();
    });
  });

  describe('Click Handler Tests', () => {
    test('should call onDetailsClick when details button is clicked', () => {
      const onDetailsClick = vi.fn();
      render(
        <RecommendationCard
          recommendation={mockRecommendation}
          onDetailsClick={onDetailsClick}
        />
      );

      const button = screen.getByText('Detaylar');
      fireEvent.click(button);

      expect(onDetailsClick).toHaveBeenCalledTimes(1);
    });

    test('should call onTrendyolClick when Trendyol button is clicked', () => {
      const onTrendyolClick = vi.fn();
      render(
        <RecommendationCard
          recommendation={mockRecommendation}
          onTrendyolClick={onTrendyolClick}
        />
      );

      const button = screen.getByText("Trendyol'da Gör");
      fireEvent.click(button);

      expect(onTrendyolClick).toHaveBeenCalledTimes(1);
    });

    test('should open Trendyol URL in new tab when onTrendyolClick is not provided', () => {
      const windowOpenSpy = vi.spyOn(window, 'open').mockImplementation(() => null);

      render(<RecommendationCard recommendation={mockRecommendation} />);

      const button = screen.getByText("Trendyol'da Gör");
      fireEvent.click(button);

      expect(windowOpenSpy).toHaveBeenCalledWith(
        'https://trendyol.com/product/123',
        '_blank',
        'noopener,noreferrer'
      );

      windowOpenSpy.mockRestore();
    });
  });

  describe('Responsive Design Tests', () => {
    test('should have responsive classes for mobile and desktop layouts', () => {
      const { container } = render(<RecommendationCard recommendation={mockRecommendation} />);

      // Check for responsive flex classes on button container
      const buttonContainers = container.querySelectorAll('.flex.flex-col');
      expect(buttonContainers.length).toBeGreaterThan(0);
      
      // Check that at least one has responsive classes in className
      const hasResponsiveClass = Array.from(buttonContainers).some(el => 
        el.className.includes('sm:flex-row') || el.className.includes('gap')
      );
      expect(hasResponsiveClass).toBe(true);
    });

    test('should apply hover effects on card', () => {
      const { container } = render(<RecommendationCard recommendation={mockRecommendation} />);

      const card = container.querySelector('.hover\\:shadow-lg');
      expect(card).toBeTruthy();
    });

    test('should apply hover effects on image', () => {
      const { container } = render(<RecommendationCard recommendation={mockRecommendation} />);

      const image = container.querySelector('.hover\\:scale-105');
      expect(image).toBeTruthy();
    });
  });

  describe('Price Formatting Tests', () => {
    test('should format price correctly in Turkish Lira', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      // Turkish Lira format: ₺299,99
      expect(screen.getByText('₺299,99')).toBeTruthy();
    });

    test('should format large prices correctly', () => {
      const expensiveRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          price: 12345.67,
        },
      };

      render(<RecommendationCard recommendation={expensiveRec} />);

      expect(screen.getByText('₺12.345,67')).toBeTruthy();
    });

    test('should format small prices correctly', () => {
      const cheapRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          price: 9.99,
        },
      };

      render(<RecommendationCard recommendation={cheapRec} />);

      expect(screen.getByText('₺9,99')).toBeTruthy();
    });
  });

  describe('Rating Display Tests', () => {
    test('should display correct number of full stars for whole number rating', () => {
      const { container } = render(<RecommendationCard recommendation={mockRecommendation} />);

      // Rating is 4.5, should have 4 full stars and 1 half star
      const fullStars = container.querySelectorAll('svg.text-yellow-400.fill-current');
      expect(fullStars.length).toBe(4);
    });

    test('should display rating value with one decimal place', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      expect(screen.getByText('(4.5)')).toBeTruthy();
    });

    test('should handle perfect 5.0 rating', () => {
      const perfectRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          rating: 5.0,
        },
      };

      const { container } = render(<RecommendationCard recommendation={perfectRec} />);

      const fullStars = container.querySelectorAll('svg.text-yellow-400.fill-current');
      expect(fullStars.length).toBe(5);
      expect(screen.getByText('(5.0)')).toBeTruthy();
    });

    test('should handle low rating', () => {
      const lowRatingRec: GiftRecommendation = {
        ...mockRecommendation,
        gift: {
          ...mockRecommendation.gift,
          rating: 1.5,
        },
      };

      const { container } = render(<RecommendationCard recommendation={lowRatingRec} />);

      const fullStars = container.querySelectorAll('svg.text-yellow-400.fill-current');
      expect(fullStars.length).toBe(1);
      expect(screen.getByText('(1.5)')).toBeTruthy();
    });
  });

  describe('Confidence Score Display Tests', () => {
    test('should display confidence score as percentage', () => {
      render(<RecommendationCard recommendation={mockRecommendation} />);

      expect(screen.getByText('85%')).toBeTruthy();
    });

    test('should round confidence score to nearest integer', () => {
      const rec: GiftRecommendation = {
        ...mockRecommendation,
        confidenceScore: 0.876,
      };

      render(<RecommendationCard recommendation={rec} />);

      expect(screen.getByText('88%')).toBeTruthy();
    });

    test('should handle 100% confidence', () => {
      const perfectConfidenceRec: GiftRecommendation = {
        ...mockRecommendation,
        confidenceScore: 1.0,
      };

      render(<RecommendationCard recommendation={perfectConfidenceRec} />);

      expect(screen.getByText('100%')).toBeTruthy();
    });

    test('should handle 0% confidence', () => {
      const zeroConfidenceRec: GiftRecommendation = {
        ...mockRecommendation,
        confidenceScore: 0.0,
      };

      render(<RecommendationCard recommendation={zeroConfidenceRec} />);

      expect(screen.getByText('0%')).toBeTruthy();
    });
  });
});
