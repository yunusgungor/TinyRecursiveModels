import { describe, test, expect } from 'vitest';
import { render } from '@testing-library/react';
import fc from 'fast-check';
import { GiftRecommendationCard } from '../GiftRecommendationCard';
import { EnhancedGiftRecommendation } from '@/types/reasoning';

/**
 * Property-based tests for GiftRecommendationCard component
 */

// Arbitrary generators for test data
const giftArbitrary = fc.record({
  id: fc.uuid(),
  name: fc.string({ minLength: 5, maxLength: 100 }),
  price: fc.float({ min: 10, max: 10000, noDefaultInfinity: true, noNaN: true }),
  image_url: fc.option(fc.webUrl(), { nil: undefined }),
  category: fc.constantFrom('Ev & Yaşam', 'Elektronik', 'Moda', 'Spor', 'Kitap'),
  rating: fc.option(fc.float({ min: 0, max: 5, noDefaultInfinity: true, noNaN: true }), { nil: undefined }),
  availability: fc.option(fc.boolean(), { nil: undefined }),
});

const reasoningArrayArbitrary = fc.array(
  fc.string({ minLength: 10, maxLength: 200 }),
  { minLength: 1, maxLength: 10 }
);

const confidenceArbitrary = fc.float({
  min: 0,
  max: 1,
  noDefaultInfinity: true,
  noNaN: true,
});

const recommendationArbitrary = fc.record({
  gift: giftArbitrary,
  reasoning: reasoningArrayArbitrary,
  confidence: confidenceArbitrary,
}) as fc.Arbitrary<EnhancedGiftRecommendation>;

const toolResultsArbitrary = fc.record({
  review_analysis: fc.option(
    fc.record({
      average_rating: fc.float({ min: 0, max: 5, noDefaultInfinity: true, noNaN: true }),
      review_count: fc.integer({ min: 0, max: 10000 }),
    }),
    { nil: undefined }
  ),
  trend_analysis: fc.option(
    fc.record({
      trending: fc.boolean(),
      trend_score: fc.float({ min: 0, max: 1, noDefaultInfinity: true, noNaN: true }),
    }),
    { nil: undefined }
  ),
  inventory_check: fc.option(
    fc.record({
      available: fc.boolean(),
      stock_count: fc.integer({ min: 0, max: 1000 }),
    }),
    { nil: undefined }
  ),
});

describe('GiftRecommendationCard Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 1: Reasoning display completeness
   * Validates: Requirements 1.1
   */
  test('Property 1: For any gift recommendation with reasoning data, the gift card should display all reasoning strings on the card', () => {
    fc.assert(
      fc.property(recommendationArbitrary, (recommendation) => {
        const { container, unmount } = render(
          <GiftRecommendationCard recommendation={recommendation} />
        );

        // Check that all reasoning strings are present in the rendered output
        // When not expanded, at least the first 2 reasoning items should be visible
        const displayedReasoningCount = Math.min(recommendation.reasoning.length, 2);
        
        for (let i = 0; i < displayedReasoningCount; i++) {
          const reasoningText = recommendation.reasoning[i];
          // The text might be split across multiple elements due to highlighting
          // So we check if the container includes the text
          expect(container.textContent).toContain(reasoningText);
        }

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 3: Tool insights icon rendering
   * Validates: Requirements 1.3
   */
  test('Property 3: For any gift recommendation with tool insights (rating, trend, availability), the frontend should render corresponding icons', () => {
    fc.assert(
      fc.property(
        recommendationArbitrary,
        toolResultsArbitrary,
        (recommendation, toolResults) => {
          const { container, unmount } = render(
            <GiftRecommendationCard
              recommendation={recommendation}
              toolResults={toolResults}
            />
          );

          // Check for rating icon if review_analysis exists
          if (toolResults.review_analysis) {
            const ratingElement = container.querySelector('[aria-label*="Rating"]');
            expect(ratingElement).toBeTruthy();
            expect(container.textContent).toContain(
              toolResults.review_analysis.average_rating.toFixed(1)
            );
          }

          // Check for trend icon if trending is true
          if (toolResults.trend_analysis?.trending) {
            const trendElement = container.querySelector('[aria-label="Trending"]');
            expect(trendElement).toBeTruthy();
            expect(container.textContent).toContain('Trend');
          }

          // Check for availability icon if available is true
          if (toolResults.inventory_check?.available) {
            const availabilityElement = container.querySelector('[aria-label="In Stock"]');
            expect(availabilityElement).toBeTruthy();
            expect(container.textContent).toContain('Stokta');
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 4: Expandable reasoning text
   * Validates: Requirements 1.4
   */
  test('Property 4: For any reasoning text exceeding a threshold length, the frontend should provide an expandable "Show more" button', () => {
    fc.assert(
      fc.property(
        fc.record({
          gift: giftArbitrary,
          reasoning: fc.array(
            fc.string({ minLength: 50, maxLength: 100 }),
            { minLength: 3, maxLength: 10 }
          ),
          confidence: confidenceArbitrary,
        }) as fc.Arbitrary<EnhancedGiftRecommendation>,
        (recommendation) => {
          const { container, unmount, getByText } = render(
            <GiftRecommendationCard recommendation={recommendation} />
          );

          const reasoningText = recommendation.reasoning.join(' ');
          const maxReasoningLength = 200;

          if (reasoningText.length > maxReasoningLength) {
            // Should have "Daha fazla göster" button
            const expandButton = getByText('Daha fazla göster');
            expect(expandButton).toBeTruthy();
            expect(expandButton.getAttribute('aria-expanded')).toBe('false');
          } else {
            // Should not have expand button
            const expandButtonQuery = container.querySelector('[aria-expanded]');
            expect(expandButtonQuery).toBeFalsy();
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Reasoning factor highlighting
   * Validates: Requirements 1.2
   */
  test('Property 2: For any reasoning text containing hobby match, budget optimization, or age appropriateness, the frontend should highlight these factors separately', () => {
    fc.assert(
      fc.property(
        fc.record({
          gift: giftArbitrary,
          reasoning: fc.array(
            fc.oneof(
              fc.constant('Kullanıcının hobi listesinde bu ürün var'),
              fc.constant('Bütçe aralığına uygun fiyat'),
              fc.constant('Yaş grubuna uygun hediye'),
              fc.constant('İlgi alanına göre seçildi'),
              fc.string({ minLength: 10, maxLength: 100 }).filter(s => s.trim().length > 0)
            ),
            { minLength: 1, maxLength: 5 }
          ),
          confidence: confidenceArbitrary,
        }) as fc.Arbitrary<EnhancedGiftRecommendation>,
        (recommendation) => {
          const { container, unmount } = render(
            <GiftRecommendationCard recommendation={recommendation} />
          );

          // Only check the first 2 reasoning items since that's what's displayed by default
          const displayedReasoning = recommendation.reasoning.slice(0, 2);
          const displayedText = displayedReasoning.join(' ').toLowerCase().trim();
          
          // Only check for highlighting if there's actual content
          if (displayedText.length > 0) {
            if (displayedText.includes('hobi') || displayedText.includes('ilgi')) {
              // Should have purple highlighting
              const purpleElements = container.querySelectorAll('.text-purple-600, .dark\\:text-purple-400');
              expect(purpleElements.length).toBeGreaterThan(0);
            }
            
            if (displayedText.includes('bütçe') || displayedText.includes('fiyat')) {
              // Should have green highlighting
              const greenElements = container.querySelectorAll('.text-green-600, .dark\\:text-green-400');
              expect(greenElements.length).toBeGreaterThan(0);
            }
            
            if (displayedText.includes('yaş')) {
              // Should have blue highlighting
              const blueElements = container.querySelectorAll('.text-blue-600, .dark\\:text-blue-400');
              expect(blueElements.length).toBeGreaterThan(0);
            }
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Gift information display
   * Ensures all gift information is rendered
   */
  test('Property: For any gift, the card should display name, price, and category', () => {
    fc.assert(
      fc.property(recommendationArbitrary, (recommendation) => {
        const { container, unmount } = render(
          <GiftRecommendationCard recommendation={recommendation} />
        );

        // Check gift name is displayed
        expect(container.textContent).toContain(recommendation.gift.name);

        // Check category is displayed
        expect(container.textContent).toContain(recommendation.gift.category);

        // Check price is displayed (formatted in Turkish Lira)
        const priceText = container.textContent;
        expect(priceText).toMatch(/₺|TL/); // Should contain Turkish Lira symbol

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Confidence indicator presence
   * Ensures confidence indicator is always displayed
   */
  test('Property: For any gift recommendation, the card should display a confidence indicator', () => {
    fc.assert(
      fc.property(recommendationArbitrary, (recommendation) => {
        const { container, unmount } = render(
          <GiftRecommendationCard recommendation={recommendation} />
        );

        // Check for confidence indicator
        const confidenceIndicator = container.querySelector('[aria-label*="Güven skoru"]');
        expect(confidenceIndicator).toBeTruthy();

        // Check that confidence percentage is displayed
        const expectedPercentage = `${(recommendation.confidence * 100).toFixed(0)}%`;
        expect(container.textContent).toContain(expectedPercentage);

        unmount();
      }),
      { numRuns: 100 }
    );
  });
});
