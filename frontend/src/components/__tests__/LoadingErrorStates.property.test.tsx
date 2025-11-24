/**
 * Property-based tests for loading and error state components
 * 
 * **Feature: frontend-reasoning-visualization**
 * Tests Properties 36, 37, 38
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import * as fc from 'fast-check';
import {
  Spinner,
  GiftCardSkeleton,
  ToolSelectionSkeleton,
  CategoryChartSkeleton,
  AttentionWeightsSkeleton,
  ThinkingStepsSkeleton,
  ReasoningPanelSkeleton,
  LoadingOverlay,
} from '../LoadingStates';
import { ErrorMessage, InlineErrorMessage, NetworkError, TimeoutError } from '../ErrorStates';

describe('Loading and Error States - Property Tests', () => {
  /**
   * **Feature: frontend-reasoning-visualization, Property 36: Loading state display**
   * **Validates: Requirements 9.1**
   * 
   * For any reasoning data loading, the frontend should display a skeleton loader or spinner.
   */
  describe('Property 36: Loading state display', () => {
    it('should display loading indicator with proper ARIA attributes for any loading state', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(
            'spinner-sm',
            'spinner-md',
            'spinner-lg',
            'gift-card',
            'tool-selection',
            'category-chart',
            'attention-weights',
            'thinking-steps',
            'reasoning-panel',
            'overlay'
          ),
          (loaderType) => {
            let component;
            
            if (loaderType === 'spinner-sm') component = <Spinner size="sm" />;
            else if (loaderType === 'spinner-md') component = <Spinner size="md" />;
            else if (loaderType === 'spinner-lg') component = <Spinner size="lg" />;
            else if (loaderType === 'gift-card') component = <GiftCardSkeleton />;
            else if (loaderType === 'tool-selection') component = <ToolSelectionSkeleton />;
            else if (loaderType === 'category-chart') component = <CategoryChartSkeleton />;
            else if (loaderType === 'attention-weights') component = <AttentionWeightsSkeleton />;
            else if (loaderType === 'thinking-steps') component = <ThinkingStepsSkeleton />;
            else if (loaderType === 'reasoning-panel') component = <ReasoningPanelSkeleton />;
            else component = <LoadingOverlay />;

            const { container, unmount } = render(component);

            // All loading states should have role="status"
            const statusElements = container.querySelectorAll('[role="status"]');
            expect(statusElements.length).toBeGreaterThan(0);

            // All loading states should have aria-label
            const ariaLabelElements = container.querySelectorAll('[aria-label]');
            expect(ariaLabelElements.length).toBeGreaterThan(0);

            // Verify loading indicator is visible
            const loadingText = screen.queryAllByText(/loading|yükleniyor/i);
            expect(loadingText.length > 0 || statusElements.length > 0).toBeTruthy();

            // Cleanup
            unmount();
          }
        ),
        { numRuns: 100 }
      );
    });

    it('should display spinner with correct size class for any size variant', () => {
      fc.assert(
        fc.property(fc.constantFrom('sm', 'md', 'lg'), (size) => {
          const { container } = render(<Spinner size={size} />);

          const spinner = container.querySelector('[role="status"]');
          expect(spinner).toBeTruthy();

          // Verify size-specific classes are applied
          const sizeClasses = {
            sm: 'w-4',
            md: 'w-8',
            lg: 'w-12',
          };

          expect(spinner?.className).toContain(sizeClasses[size]);
        }),
        { numRuns: 100 }
      );
    });

    it('should display skeleton loaders with animation for any component type', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(
            'gift-card',
            'tool-selection',
            'category-chart',
            'attention-weights',
            'thinking-steps'
          ),
          (componentType) => {
            const { container } = render(
              <div>
                {componentType === 'gift-card' && <GiftCardSkeleton />}
                {componentType === 'tool-selection' && <ToolSelectionSkeleton />}
                {componentType === 'category-chart' && <CategoryChartSkeleton />}
                {componentType === 'attention-weights' && <AttentionWeightsSkeleton />}
                {componentType === 'thinking-steps' && <ThinkingStepsSkeleton />}
              </div>
            );

            // All skeletons should have animate-pulse class
            const animatedElements = container.querySelectorAll('.animate-pulse');
            expect(animatedElements.length).toBeGreaterThan(0);

            // All skeletons should have role="status"
            const statusElement = container.querySelector('[role="status"]');
            expect(statusElement).toBeTruthy();
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: frontend-reasoning-visualization, Property 37: Error message display**
   * **Validates: Requirements 9.2**
   * 
   * For any API request failure, the frontend should display a user-friendly error message.
   */
  describe('Property 37: Error message display', () => {
    it('should display error message with proper ARIA attributes for any error', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 1, maxLength: 200 }).filter(s => s.trim().length > 0),
          fc.option(fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0), { nil: undefined }),
          (errorMessage, title) => {
            const { container, unmount } = render(
              <ErrorMessage error={errorMessage} title={title} />
            );

            // Error should have role="alert"
            const alertElement = container.querySelector('[role="alert"]');
            expect(alertElement).toBeTruthy();

            // Error should have aria-live="assertive"
            const assertiveElement = container.querySelector('[aria-live="assertive"]');
            expect(assertiveElement).toBeTruthy();

            // Error message should be displayed
            const errorElements = container.querySelectorAll('p');
            const hasErrorMessage = Array.from(errorElements).some(el => 
              el.textContent === errorMessage
            );
            expect(hasErrorMessage).toBeTruthy();

            // Title should be displayed (default or custom)
            const displayedTitle = title || 'Bir Hata Oluştu';
            const titleElements = container.querySelectorAll('h3');
            const hasTitle = Array.from(titleElements).some(el => 
              el.textContent === displayedTitle
            );
            expect(hasTitle).toBeTruthy();

            // Cleanup
            unmount();
          }
        ),
        { numRuns: 100 }
      );
    });

    it('should display error icon for any error message', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 1, maxLength: 200 }).filter(s => s.trim().length > 0),
          (errorMessage) => {
            const { container, unmount } = render(<ErrorMessage error={errorMessage} />);

            // Error icon should be present (SVG with aria-hidden)
            const errorIcon = container.querySelector('svg[aria-hidden="true"]');
            expect(errorIcon).toBeTruthy();

            // Icon should have appropriate color class
            const className = errorIcon?.getAttribute('class') || '';
            expect(className).toMatch(/text-red/);

            // Cleanup
            unmount();
          }
        ),
        { numRuns: 100 }
      );
    });

    it('should display inline error message for any short error text', () => {
      fc.assert(
        fc.property(
          fc.string({ minLength: 1, maxLength: 100 }).filter(s => s.trim().length > 0),
          (message) => {
            const { container, unmount } = render(<InlineErrorMessage message={message} />);

            // Should have role="alert"
            const alertElement = container.querySelector('[role="alert"]');
            expect(alertElement).toBeTruthy();

            // Message should be displayed
            const spans = container.querySelectorAll('span');
            const hasMessage = Array.from(spans).some(el => el.textContent === message);
            expect(hasMessage).toBeTruthy();

            // Should have error styling
            const className = alertElement?.getAttribute('class') || '';
            expect(className).toMatch(/text-red/);

            // Cleanup
            unmount();
          }
        ),
        { numRuns: 100 }
      );
    });

    it('should display specific error types with appropriate messages', () => {
      fc.assert(
        fc.property(fc.constantFrom('network', 'timeout'), (errorType) => {
          const component = errorType === 'network' ? <NetworkError /> : <TimeoutError />;
          const { container, unmount } = render(component);

          // Should have role="alert"
          const alertElement = container.querySelector('[role="alert"]');
          expect(alertElement).toBeTruthy();

          // Should display appropriate error message
          if (errorType === 'network') {
            const elements = screen.getAllByText(/bağlantı|internet/i);
            expect(elements.length).toBeGreaterThan(0);
          } else if (errorType === 'timeout') {
            const elements = screen.getAllByText(/zaman aşımı/i);
            expect(elements.length).toBeGreaterThan(0);
          }

          // Cleanup
          unmount();
        }),
        { numRuns: 100 }
      );
    });
  });

  /**
   * **Feature: frontend-reasoning-visualization, Property 38: Retry button display**
   * **Validates: Requirements 9.4**
   * 
   * For any error state with retry capability, the frontend should display a "Retry" button.
   */
  describe('Property 38: Retry button display', () => {
    it('should display retry button when onRetry callback is provided', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 200 }), (errorMessage) => {
          const onRetry = () => {};
          const { unmount } = render(<ErrorMessage error={errorMessage} onRetry={onRetry} />);

          // Retry button should be present
          const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
          expect(retryButton).toBeInTheDocument();

          // Button should have appropriate styling
          expect(retryButton.className).toMatch(/bg-red/);

          // Button should have aria-label
          expect(retryButton).toHaveAttribute('aria-label', 'Tekrar dene');

          // Cleanup
          unmount();
        }),
        { numRuns: 100 }
      );
    });

    it('should not display retry button when onRetry callback is not provided', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 200 }), (errorMessage) => {
          const { unmount } = render(<ErrorMessage error={errorMessage} />);

          // Retry button should not be present
          const retryButton = screen.queryByRole('button', { name: /tekrar dene/i });
          expect(retryButton).not.toBeInTheDocument();

          // Cleanup
          unmount();
        }),
        { numRuns: 100 }
      );
    });

    it('should display retry button with icon for any error type with retry', () => {
      fc.assert(
        fc.property(fc.constantFrom('network', 'timeout', 'generic'), (errorType) => {
          const onRetry = () => {};
          let component;
          
          if (errorType === 'network') {
            component = <NetworkError onRetry={onRetry} />;
          } else if (errorType === 'timeout') {
            component = <TimeoutError onRetry={onRetry} />;
          } else {
            component = <ErrorMessage error="Generic error" onRetry={onRetry} />;
          }

          const { unmount } = render(component);

          // Retry button should be present
          const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
          expect(retryButton).toBeInTheDocument();

          // Button should contain an icon (SVG)
          const icon = retryButton.querySelector('svg');
          expect(icon).toBeTruthy();

          // Cleanup
          unmount();
        }),
        { numRuns: 100 }
      );
    });

    it('should display retry button with proper focus management', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 200 }), (errorMessage) => {
          const onRetry = () => {};
          const { unmount } = render(<ErrorMessage error={errorMessage} onRetry={onRetry} />);

          const retryButton = screen.getByRole('button', { name: /tekrar dene/i });

          // Button should be focusable
          expect(retryButton).toHaveAttribute('class');
          expect(retryButton.className).toMatch(/focus:ring/);

          // Cleanup
          unmount();
        }),
        { numRuns: 100 }
      );
    });
  });
});
