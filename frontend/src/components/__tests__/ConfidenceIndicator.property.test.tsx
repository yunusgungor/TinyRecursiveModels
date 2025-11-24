import { describe, test, expect } from 'vitest';
import { render } from '@testing-library/react';
import fc from 'fast-check';
import { ConfidenceIndicator } from '../ConfidenceIndicator';

/**
 * Property-based tests for ConfidenceIndicator component
 */

describe('ConfidenceIndicator Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 27: High confidence styling
   * Validates: Requirements 6.2
   */
  test('Property 27: For any confidence score above 0.8, the frontend should display green color and "Yüksek Güven" label', () => {
    fc.assert(
      fc.property(
        fc.float({ min: Math.fround(0.8), max: Math.fround(1.0), noDefaultInfinity: true, noNaN: true }).filter(n => n > 0.8),
        (confidence) => {
          const { container, unmount } = render(
            <ConfidenceIndicator confidence={confidence} />
          );

          // Check for green color classes
          const indicator = container.querySelector('[role="status"]');
          expect(indicator).toBeTruthy();
          
          const classList = indicator?.className || '';
          expect(classList).toContain('bg-green-100');
          expect(classList).toContain('text-green-800');
          expect(classList).toContain('border-green-300');

          // Check for "Yüksek Güven" label
          expect(container.textContent).toContain('Yüksek Güven');

          // Check for percentage display
          const expectedPercentage = `${(confidence * 100).toFixed(0)}%`;
          expect(container.textContent).toContain(expectedPercentage);

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 28: Medium confidence styling
   * Validates: Requirements 6.3
   */
  test('Property 28: For any confidence score between 0.5 and 0.8, the frontend should display yellow color and "Orta Güven" label', () => {
    fc.assert(
      fc.property(
        fc.float({ min: Math.fround(0.5), max: Math.fround(0.8), noDefaultInfinity: true, noNaN: true }),
        (confidence) => {
          const { container, unmount } = render(
            <ConfidenceIndicator confidence={confidence} />
          );

          // Check for yellow color classes
          const indicator = container.querySelector('[role="status"]');
          expect(indicator).toBeTruthy();
          
          const classList = indicator?.className || '';
          expect(classList).toContain('bg-yellow-100');
          expect(classList).toContain('text-yellow-800');
          expect(classList).toContain('border-yellow-300');

          // Check for "Orta Güven" label
          expect(container.textContent).toContain('Orta Güven');

          // Check for percentage display
          const expectedPercentage = `${(confidence * 100).toFixed(0)}%`;
          expect(container.textContent).toContain(expectedPercentage);

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 29: Low confidence styling
   * Validates: Requirements 6.4
   */
  test('Property 29: For any confidence score below 0.5, the frontend should display red color and "Düşük Güven" label', () => {
    fc.assert(
      fc.property(
        fc.float({ min: Math.fround(0.0), max: Math.fround(0.5), noDefaultInfinity: true, noNaN: true }).filter(n => n < 0.5),
        (confidence) => {
          const { container, unmount } = render(
            <ConfidenceIndicator confidence={confidence} />
          );

          // Check for red color classes
          const indicator = container.querySelector('[role="status"]');
          expect(indicator).toBeTruthy();
          
          const classList = indicator?.className || '';
          expect(classList).toContain('bg-red-100');
          expect(classList).toContain('text-red-800');
          expect(classList).toContain('border-red-300');

          // Check for "Düşük Güven" label
          expect(container.textContent).toContain('Düşük Güven');

          // Check for percentage display
          const expectedPercentage = `${(confidence * 100).toFixed(0)}%`;
          expect(container.textContent).toContain(expectedPercentage);

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
