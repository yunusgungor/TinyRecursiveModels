import { describe, test, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import fc from 'fast-check';
import { ConfidenceExplanationModal } from '../ConfidenceExplanationModal';
import { ConfidenceExplanation } from '@/types';

/**
 * Property-based tests for ConfidenceExplanationModal component
 */

describe('ConfidenceExplanationModal Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 30: Confidence explanation modal
   * Validates: Requirements 6.5
   */
  test('Property 30: For any confidence indicator click, the frontend should display confidence explanation (positive and negative factors) in a modal', () => {
    fc.assert(
      fc.property(
        // Generate random confidence score
        fc.float({ min: 0.0, max: 1.0, noDefaultInfinity: true, noNaN: true }),
        // Generate random positive factors (0-10 factors) - use alphanumeric strings
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,100}$/),
          { minLength: 0, maxLength: 10 }
        ),
        // Generate random negative factors (0-10 factors) - use alphanumeric strings
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,100}$/),
          { minLength: 0, maxLength: 10 }
        ),
        (score, positiveFactors, negativeFactors) => {
          // Determine level based on score
          const level: 'high' | 'medium' | 'low' = 
            score > 0.8 ? 'high' : score >= 0.5 ? 'medium' : 'low';

          const explanation: ConfidenceExplanation = {
            score,
            level,
            factors: {
              positive: positiveFactors,
              negative: negativeFactors,
            },
          };

          const { container, unmount } = render(
            <ConfidenceExplanationModal
              isOpen={true}
              onClose={() => {}}
              explanation={explanation}
            />
          );

          try {
            // Verify modal is displayed
            const modal = screen.getByRole('dialog');
            expect(modal).toBeTruthy();

            // Verify title is present (modal is in a portal, so check document.body)
            expect(document.body.textContent).toContain('Güven Skoru Açıklaması');

            // Verify confidence score is displayed
            const expectedPercentage = `${(score * 100).toFixed(0)}%`;
            expect(document.body.textContent).toContain(expectedPercentage);

            // Verify level label is displayed
            const levelLabels = {
              high: 'Yüksek Güven',
              medium: 'Orta Güven',
              low: 'Düşük Güven',
            };
            expect(document.body.textContent).toContain(levelLabels[level]);

            // Verify positive factors section exists if there are positive factors
            if (positiveFactors.length > 0) {
              expect(document.body.textContent).toContain('Olumlu Faktörler');
              
              // Verify each positive factor is displayed
              positiveFactors.forEach((factor) => {
                expect(document.body.textContent).toContain(factor);
              });

              // Verify positive factors have green styling
              const positiveItems = document.body.querySelectorAll('.bg-green-50');
              expect(positiveItems.length).toBeGreaterThan(0);
            }

            // Verify negative factors section exists if there are negative factors
            if (negativeFactors.length > 0) {
              expect(document.body.textContent).toContain('Olumsuz Faktörler');
              
              // Verify each negative factor is displayed
              negativeFactors.forEach((factor) => {
                expect(document.body.textContent).toContain(factor);
              });

              // Verify negative factors have red styling
              const negativeItems = document.body.querySelectorAll('.bg-red-50');
              expect(negativeItems.length).toBeGreaterThan(0);
            }

            // Verify close button is present
            const closeButtons = screen.getAllByRole('button');
            expect(closeButtons.length).toBeGreaterThan(0);
          } finally {
            // Always unmount to prevent multiple modals
            unmount();
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 31: Factor categorization
   * Validates: Requirements 6.6
   */
  test('Property 31: For any confidence explanation display, the frontend should categorize factors as positive or negative', () => {
    fc.assert(
      fc.property(
        // Generate random confidence score
        fc.float({ min: 0.0, max: 1.0, noDefaultInfinity: true, noNaN: true }),
        // Generate at least 1 positive factor - use alphanumeric strings with unique prefix
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,100}$/).map(s => `positive_${s}`),
          { minLength: 1, maxLength: 10 }
        ),
        // Generate at least 1 negative factor - use alphanumeric strings with unique prefix
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,100}$/).map(s => `negative_${s}`),
          { minLength: 1, maxLength: 10 }
        ),
        (score, positiveFactors, negativeFactors) => {
          // Determine level based on score
          const level: 'high' | 'medium' | 'low' = 
            score > 0.8 ? 'high' : score >= 0.5 ? 'medium' : 'low';

          const explanation: ConfidenceExplanation = {
            score,
            level,
            factors: {
              positive: positiveFactors,
              negative: negativeFactors,
            },
          };

          const { container, unmount } = render(
            <ConfidenceExplanationModal
              isOpen={true}
              onClose={() => {}}
              explanation={explanation}
            />
          );

          try {
            // Verify positive factors are categorized correctly
            expect(document.body.textContent).toContain('Olumlu Faktörler');
            
            // Verify positive factors have green styling (bg-green-50)
            const positiveItems = document.body.querySelectorAll('.bg-green-50');
            expect(positiveItems.length).toBe(positiveFactors.length);

            // Verify each positive factor is in a green-styled container
            positiveFactors.forEach((factor) => {
              const factorElements = Array.from(document.body.querySelectorAll('.bg-green-50'))
                .filter(el => el.textContent?.includes(factor));
              expect(factorElements.length).toBeGreaterThan(0);
            });

            // Verify negative factors are categorized correctly
            expect(document.body.textContent).toContain('Olumsuz Faktörler');
            
            // Verify negative factors have red styling (bg-red-50)
            const negativeItems = document.body.querySelectorAll('.bg-red-50');
            expect(negativeItems.length).toBe(negativeFactors.length);

            // Verify each negative factor is in a red-styled container
            negativeFactors.forEach((factor) => {
              const factorElements = Array.from(document.body.querySelectorAll('.bg-red-50'))
                .filter(el => el.textContent?.includes(factor));
              expect(factorElements.length).toBeGreaterThan(0);
            });

            // Verify positive and negative sections are separate
            // (positive factors should not be in red containers and vice versa)
            positiveFactors.forEach((factor) => {
              const inRedContainer = Array.from(document.body.querySelectorAll('.bg-red-50'))
                .some(el => el.textContent?.includes(factor));
              expect(inRedContainer).toBe(false);
            });

            negativeFactors.forEach((factor) => {
              const inGreenContainer = Array.from(document.body.querySelectorAll('.bg-green-50'))
                .some(el => el.textContent?.includes(factor));
              expect(inGreenContainer).toBe(false);
            });
          } finally {
            // Always unmount to prevent multiple modals
            unmount();
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property test: Modal accessibility
   * Verifies keyboard navigation support
   */
  test('Property: Modal should have proper ARIA attributes for accessibility', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0.0, max: 1.0, noDefaultInfinity: true, noNaN: true }),
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,50}$/),
          { minLength: 1, maxLength: 5 }
        ),
        fc.array(
          fc.stringMatching(/^[a-zA-Z0-9 ]{5,50}$/),
          { minLength: 1, maxLength: 5 }
        ),
        (score, positiveFactors, negativeFactors) => {
          const level: 'high' | 'medium' | 'low' = 
            score > 0.8 ? 'high' : score >= 0.5 ? 'medium' : 'low';

          const explanation: ConfidenceExplanation = {
            score,
            level,
            factors: {
              positive: positiveFactors,
              negative: negativeFactors,
            },
          };

          const { unmount } = render(
            <ConfidenceExplanationModal
              isOpen={true}
              onClose={() => {}}
              explanation={explanation}
            />
          );

          try {
            // Verify modal has dialog role
            const modal = screen.getByRole('dialog');
            expect(modal).toBeTruthy();

            // Verify modal has aria-describedby
            expect(modal.getAttribute('aria-describedby')).toBe('confidence-explanation-description');

            // Verify lists have proper roles
            const lists = screen.getAllByRole('list');
            expect(lists.length).toBeGreaterThan(0);

            // Verify close buttons have proper labels
            const closeButtons = screen.getAllByRole('button');
            const hasLabeledCloseButton = closeButtons.some(
              button => button.getAttribute('aria-label') === 'Kapat' || button.textContent === 'Kapat'
            );
            expect(hasLabeledCloseButton).toBe(true);
          } finally {
            // Always unmount to prevent multiple modals
            unmount();
          }
        }
      ),
      { numRuns: 100 }
    );
  });
});
