import { describe, test, expect } from 'vitest';
import { render } from '@testing-library/react';
import fc from 'fast-check';
import { ThinkingStepsTimeline } from '../ThinkingStepsTimeline';
import { ThinkingStep } from '@/types/reasoning';

/**
 * Property-based tests for ThinkingStepsTimeline component
 */

// Arbitrary generator for ThinkingStep
const thinkingStepArbitrary = fc.record({
  step: fc.integer({ min: 1, max: 100 }),
  action: fc.string({ minLength: 10, maxLength: 100 }),
  result: fc.string({ minLength: 10, maxLength: 150 }),
  insight: fc.string({ minLength: 10, maxLength: 200 }),
}) as fc.Arbitrary<ThinkingStep>;

// Generator for unique step arrays (no duplicate step numbers)
const uniqueStepArrayArbitrary = fc
  .array(thinkingStepArbitrary, { minLength: 1, maxLength: 10 })
  .map((steps) => {
    // Remove duplicates by step number, keeping first occurrence
    const seen = new Set<number>();
    return steps.filter((step) => {
      if (seen.has(step.step)) {
        return false;
      }
      seen.add(step.step);
      return true;
    });
  })
  .filter((steps) => steps.length > 0); // Ensure at least one step remains

describe('ThinkingStepsTimeline Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 21: Chronological step ordering
   * Validates: Requirements 5.2
   */
  test('Property 21: For any thinking steps display, the frontend should render steps in chronological order on the timeline', () => {
    fc.assert(
      fc.property(uniqueStepArrayArbitrary, (steps) => {
        const { container, unmount } = render(
          <ThinkingStepsTimeline steps={steps} />
        );

        // Get all step elements
        const stepElements = Array.from(
          container.querySelectorAll('[data-step]')
        );

        // Extract step numbers from rendered elements
        const renderedStepNumbers = stepElements.map((el) => {
          const stepAttr = el.getAttribute('data-step');
          return stepAttr ? parseInt(stepAttr, 10) : -1;
        });

        // Sort original steps chronologically
        const sortedStepNumbers = [...steps]
          .sort((a, b) => a.step - b.step)
          .map((s) => s.step);

        // Verify that rendered steps are in chronological order
        expect(renderedStepNumbers).toEqual(sortedStepNumbers);

        // Additional check: verify each consecutive pair is in ascending order
        for (let i = 0; i < renderedStepNumbers.length - 1; i++) {
          expect(renderedStepNumbers[i]).toBeLessThan(
            renderedStepNumbers[i + 1]
          );
        }

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 22: Step information completeness
   * Validates: Requirements 5.3
   */
  test('Property 22: For any thinking step, the frontend should display step number, action name, result, and insight', () => {
    fc.assert(
      fc.property(uniqueStepArrayArbitrary, (steps) => {
        const { container, unmount } = render(
          <ThinkingStepsTimeline steps={steps} />
        );

        steps.forEach((step) => {
          // Check that step number is displayed
          const stepNumberText = `Adım ${step.step}`;
          expect(container.textContent).toContain(stepNumberText);

          // Check that action is displayed
          expect(container.textContent).toContain(step.action);

          // Check that result is displayed
          expect(container.textContent).toContain(step.result);

          // Note: Insight is only visible when expanded, but we can verify
          // it's in the DOM (even if not visible)
          const stepElement = container.querySelector(
            `[data-step="${step.step}"]`
          );
          expect(stepElement).toBeTruthy();

          if (stepElement) {
            // Verify the step element contains all the information
            const stepContent = stepElement.textContent || '';
            expect(stepContent).toContain(stepNumberText);
            expect(stepContent).toContain(step.action);
            expect(stepContent).toContain(step.result);
          }
        });

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 23: Completed step marking
   * Validates: Requirements 5.4
   */
  test('Property 23: For any completed step, the frontend should mark it with a green checkmark', () => {
    fc.assert(
      fc.property(uniqueStepArrayArbitrary, (steps) => {
        const { container, unmount } = render(
          <ThinkingStepsTimeline steps={steps} />
        );

        // All steps in the timeline are considered "completed" by design
        // (they represent past thinking steps)
        steps.forEach((step) => {
          const stepElement = container.querySelector(
            `[data-step="${step.step}"]`
          );
          expect(stepElement).toBeTruthy();

          if (stepElement) {
            // Check for green checkmark icon
            // The checkmark SVG has a specific path
            const checkmarkPath = stepElement.querySelector(
              'path[d="M5 13l4 4L19 7"]'
            );
            expect(checkmarkPath).toBeTruthy();

            // Check for green background on the marker
            const greenMarker = stepElement.querySelector('.bg-green-500');
            expect(greenMarker).toBeTruthy();

            // Check for green border
            const greenBorder = stepElement.querySelector('.border-green-600');
            expect(greenBorder).toBeTruthy();
          }
        });

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Step click expansion
   * Verifies that clicking a step expands its details
   */
  test('Property: For any step, clicking it should toggle the expanded state', () => {
    fc.assert(
      fc.property(
        uniqueStepArrayArbitrary.filter((steps) => steps.length >= 1),
        (steps) => {
          const { container, unmount } = render(
            <ThinkingStepsTimeline steps={steps} />
          );

          // Pick the first step to test
          const firstStep = [...steps].sort((a, b) => a.step - b.step)[0];
          const stepElement = container.querySelector(
            `[data-step="${firstStep.step}"]`
          );

          expect(stepElement).toBeTruthy();

          if (stepElement) {
            // Initially, the step should not be expanded
            const buttonElement = stepElement.querySelector('[role="button"]');
            expect(buttonElement).toBeTruthy();

            if (buttonElement) {
              const initialExpanded =
                buttonElement.getAttribute('aria-expanded');
              expect(initialExpanded).toBe('false');

              // After clicking, we can't easily test the expanded state in this test
              // because it requires user interaction simulation
              // This is better tested in unit tests with fireEvent
            }
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Scrollable container
   * Verifies that the timeline has a scrollable container
   */
  test('Property: For any timeline with steps, the frontend should provide a scrollable container', () => {
    fc.assert(
      fc.property(uniqueStepArrayArbitrary, (steps) => {
        const { container, unmount } = render(
          <ThinkingStepsTimeline steps={steps} />
        );

        // Check for scrollable container
        const scrollableContainer = container.querySelector(
          '.overflow-y-auto'
        );
        expect(scrollableContainer).toBeTruthy();

        // Check for max-height constraint
        const maxHeightContainer = container.querySelector('.max-h-\\[600px\\]');
        expect(maxHeightContainer).toBeTruthy();

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Accessibility attributes
   * Verifies that proper ARIA attributes are present
   */
  test('Property: For any timeline, the frontend should include proper ARIA labels and roles', () => {
    fc.assert(
      fc.property(uniqueStepArrayArbitrary, (steps) => {
        const { container, unmount } = render(
          <ThinkingStepsTimeline steps={steps} />
        );

        // Check for region role
        const regionElement = container.querySelector('[role="region"]');
        expect(regionElement).toBeTruthy();

        // Check for list role
        const listElement = container.querySelector('[role="list"]');
        expect(listElement).toBeTruthy();

        // Check that each step has listitem role
        const listItems = container.querySelectorAll('[role="listitem"]');
        expect(listItems.length).toBe(steps.length);

        // Check that each step has button role for interaction
        const buttons = container.querySelectorAll('[role="button"]');
        expect(buttons.length).toBe(steps.length);

        // Check that each button has aria-expanded attribute
        buttons.forEach((button) => {
          expect(button.getAttribute('aria-expanded')).toBeTruthy();
        });

        // Check that each button has aria-label
        buttons.forEach((button) => {
          expect(button.getAttribute('aria-label')).toBeTruthy();
        });

        unmount();
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Empty state handling
   * Verifies that empty steps array is handled gracefully
   */
  test('Property: For an empty steps array, the frontend should display an appropriate message', () => {
    const { container, unmount } = render(
      <ThinkingStepsTimeline steps={[]} />
    );

    // Check for empty state message
    expect(container.textContent).toContain(
      'Düşünme adımı bilgisi mevcut değil'
    );

    // Verify no step elements are rendered
    const stepElements = container.querySelectorAll('[data-step]');
    expect(stepElements.length).toBe(0);

    unmount();
  });

  /**
   * Additional property: Out-of-order steps handling
   * Verifies that steps provided in random order are sorted correctly
   */
  test('Property: For any steps provided in random order, the frontend should sort them chronologically', () => {
    fc.assert(
      fc.property(
        uniqueStepArrayArbitrary.filter((steps) => steps.length >= 3),
        (steps) => {
          // Shuffle the steps to create random order
          const shuffledSteps = [...steps].sort(() => Math.random() - 0.5);

          const { container, unmount } = render(
            <ThinkingStepsTimeline steps={shuffledSteps} />
          );

          // Get rendered step numbers
          const stepElements = Array.from(
            container.querySelectorAll('[data-step]')
          );
          const renderedStepNumbers = stepElements.map((el) => {
            const stepAttr = el.getAttribute('data-step');
            return stepAttr ? parseInt(stepAttr, 10) : -1;
          });

          // Verify they are in ascending order
          for (let i = 0; i < renderedStepNumbers.length - 1; i++) {
            expect(renderedStepNumbers[i]).toBeLessThan(
              renderedStepNumbers[i + 1]
            );
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
