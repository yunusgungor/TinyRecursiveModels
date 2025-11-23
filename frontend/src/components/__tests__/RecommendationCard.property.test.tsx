import { describe, test, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import fc from 'fast-check';
import { RecommendationCard } from '../RecommendationCard';
import { GiftRecommendation } from '@/lib/api/types';

/**
 * Property-based tests for RecommendationCard component
 */

// Arbitrary generators for test data
const giftItemArbitrary = fc.record({
  id: fc.uuid(),
  name: fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
  category: fc.constantFrom('Elektronik', 'Kitap', 'Giyim', 'Ev & Yaşam', 'Spor'),
  price: fc.float({ min: 1, max: 10000, noNaN: true }),
  rating: fc.float({ min: 0, max: 5, noNaN: true }),
  imageUrl: fc.webUrl(),
  trendyolUrl: fc.webUrl(),
  description: fc.string({ minLength: 10, maxLength: 500 }),
  tags: fc.array(fc.string({ minLength: 1, maxLength: 20 }), { minLength: 0, maxLength: 10 }),
  ageSuitability: fc.tuple(
    fc.integer({ min: 0, max: 100 }),
    fc.integer({ min: 0, max: 100 })
  ).map(([a, b]) => [Math.min(a, b), Math.max(a, b)] as [number, number]),
  occasionFit: fc.array(fc.string({ minLength: 1, maxLength: 30 }), { minLength: 0, maxLength: 5 }),
  inStock: fc.boolean(),
});

const recommendationArbitrary = fc.record({
  gift: giftItemArbitrary,
  confidenceScore: fc.float({ min: 0, max: 1, noNaN: true }),
  reasoning: fc.array(fc.string({ minLength: 10, maxLength: 100 }), { minLength: 1, maxLength: 5 }),
  toolInsights: fc.dictionary(fc.string(), fc.anything()),
  rank: fc.integer({ min: 1, max: 10 }),
});

describe('RecommendationCard Property Tests', () => {
  /**
   * Feature: trendyol-gift-recommendation-web, Property 14: Recommendation Card Rendering
   * Validates: Requirements 5.1
   */
  test('Property 14: For any list of recommendations, the UI should render exactly one card component per recommendation', () => {
    fc.assert(
      fc.property(
        fc.array(recommendationArbitrary, { minLength: 1, maxLength: 10 }),
        (recommendations) => {
          const { container } = render(
            <div>
              {recommendations.map((rec) => (
                <RecommendationCard key={rec.gift.id} recommendation={rec} />
              ))}
            </div>
          );

          // Count the number of card elements rendered
          // Each card has a specific structure with product info
          const cards = container.querySelectorAll('[class*="bg-white"][class*="rounded-lg"]');
          
          // Should render exactly one card per recommendation
          expect(cards.length).toBe(recommendations.length);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: trendyol-gift-recommendation-web, Property 15: Product Card Content Completeness
   * Validates: Requirements 5.2
   */
  test('Property 15: For any rendered product card, it should display all required fields: image, name, price, rating, and category', () => {
    fc.assert(
      fc.property(recommendationArbitrary, (recommendation) => {
        const { container, unmount } = render(
          <RecommendationCard recommendation={recommendation} />
        );

        const { gift } = recommendation;

        // Check for image
        const image = container.querySelector('img');
        expect(image).toBeTruthy();
        expect(image?.getAttribute('src')).toBe(gift.imageUrl);
        expect(image?.getAttribute('alt')).toBe(gift.name);

        // Check for name - use container.textContent to avoid multiple elements issue
        expect(container.textContent).toContain(gift.name);

        // Check for price - format as Turkish Lira
        const formattedPrice = new Intl.NumberFormat('tr-TR', {
          style: 'currency',
          currency: 'TRY',
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }).format(gift.price);
        expect(container.textContent).toContain(formattedPrice);

        // Check for rating - should display rating value
        const ratingText = `(${gift.rating.toFixed(1)})`;
        expect(container.textContent).toContain(ratingText);

        // Check for category
        expect(container.textContent).toContain(gift.category);
        
        // Clean up
        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: trendyol-gift-recommendation-web, Property 16: Low Confidence Warning Display
   * Validates: Requirements 5.6
   */
  test('Property 16: For any recommendation with confidence score below 0.5, the UI should display a warning message', () => {
    fc.assert(
      fc.property(
        recommendationArbitrary,
        (recommendation) => {
          const { container, unmount } = render(
            <RecommendationCard recommendation={recommendation} />
          );

          const isLowConfidence = recommendation.confidenceScore < 0.5;
          
          // Look for warning message
          const warningText = 'Bu öneri düşük güven skoruna sahip';
          
          if (isLowConfidence) {
            // Should display warning
            const warningElements = container.querySelectorAll('.bg-yellow-50');
            expect(warningElements.length).toBeGreaterThan(0);
            
            // Check that warning text exists in the container
            expect(container.textContent).toContain(warningText);
          } else {
            // Should NOT display warning
            const warningElements = container.querySelectorAll('.bg-yellow-50');
            expect(warningElements.length).toBe(0);
          }
          
          // Clean up
          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
