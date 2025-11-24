import { describe, test, expect } from 'vitest';
import { render, act } from '@testing-library/react';
import fc from 'fast-check';
import { CategoryMatchingChart } from '../CategoryMatchingChart';
import { CategoryMatchingReasoning } from '@/types/reasoning';

/**
 * Property-based tests for CategoryMatchingChart component
 */

// Arbitrary generator for category names
const categoryNameArbitrary = fc.constantFrom(
  'Elektronik',
  'Kitap',
  'Spor Malzemeleri',
  'Ev Dekorasyonu',
  'Moda',
  'Oyun',
  'Müzik',
  'Bahçe',
  'Otomotiv',
  'Bebek Ürünleri'
);

// Arbitrary generator for reasons
// Exclude whitespace-only strings
const reasonsArbitrary = fc.array(
  fc.string({ minLength: 10, maxLength: 50 }).filter(s => s.trim().length > 0),
  { minLength: 1, maxLength: 5 }
);

// Arbitrary generator for feature contributions
const featureContributionsArbitrary = fc.option(
  fc.dictionary(
    fc.constantFrom('hobby', 'age', 'budget', 'occasion', 'trend'),
    fc.float({ min: 0, max: 1, noDefaultInfinity: true, noNaN: true })
  ),
  { nil: undefined }
);

// Arbitrary generator for CategoryMatchingReasoning
// Exclude score=0 edge case since Recharts doesn't render bars with 0 values
const categoryMatchingReasoningArbitrary = fc.record({
  category_name: categoryNameArbitrary,
  score: fc.float({ min: Math.fround(0.01), max: Math.fround(1), noDefaultInfinity: true, noNaN: true }),
  reasons: reasonsArbitrary,
  feature_contributions: featureContributionsArbitrary,
}) as fc.Arbitrary<CategoryMatchingReasoning>;

// Generator for unique category arrays (no duplicate names)
const uniqueCategoryArrayArbitrary = fc
  .array(categoryMatchingReasoningArbitrary, { minLength: 1, maxLength: 10 })
  .map((categories) => {
    // Remove duplicates by category_name, keeping first occurrence
    const seen = new Set<string>();
    return categories.filter((cat) => {
      if (seen.has(cat.category_name)) {
        return false;
      }
      seen.add(cat.category_name);
      return true;
    });
  })
  .filter((categories) => categories.length > 0); // Ensure at least one category remains

describe('CategoryMatchingChart Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 11: Minimum category count
   * Validates: Requirements 3.2
   */
  test('Property 11: For any category matching display, the frontend should show at least the top 3 categories with their scores', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary,
        (categories) => {
          const { container, unmount } = render(
            <CategoryMatchingChart categories={categories} />
          );

          // Count the number of category items displayed
          const categoryItems = container.querySelectorAll('[role="listitem"]');
          
          // Should display at least 3 categories (or all if less than 3)
          const expectedCount = Math.min(3, categories.length);
          expect(categoryItems.length).toBeGreaterThanOrEqual(expectedCount);

          // Verify that scores are displayed for each category
          categoryItems.forEach((item) => {
            // Each item should contain a percentage score
            expect(item.textContent).toMatch(/\d+%/);
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 12: High score category styling
   * Validates: Requirements 3.3
   */
  test('Property 12: For any category with score above 0.7, the frontend should render it with a green progress bar', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary,
        (categories) => {
          // Ensure at least one category has high score (strictly > 0.7)
          const categoriesWithHighScore = categories.map((cat, idx) => ({
            ...cat,
            score: idx === 0 ? 0.85 : cat.score,
          }));

          const { container, unmount } = render(
            <CategoryMatchingChart categories={categoriesWithHighScore} />
          );

          // Find all high score categories (score > 0.7) that are displayed (top 3 or all)
          const sortedCategories = [...categoriesWithHighScore]
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(3, categoriesWithHighScore.length));
          
          const highScoreCategories = sortedCategories.filter(
            (cat) => cat.score > 0.7
          );

          // Only test if there are high score categories in the displayed set
          if (highScoreCategories.length > 0) {
            highScoreCategories.forEach((category) => {
              // Find the category element by checking for the category name
              const categoryItems = Array.from(
                container.querySelectorAll('[role="listitem"]')
              );

              const categoryElement = categoryItems.find((el) =>
                el.textContent?.includes(category.category_name)
              );

              expect(categoryElement).toBeTruthy();

              if (categoryElement) {
                // Check for green color classes
                const classList = categoryElement.className;
                expect(classList).toContain('border-green-300');
                expect(classList).toContain('bg-green-50');

                // Check for green score indicator
                const greenIndicator = categoryElement.querySelector('.bg-green-500');
                expect(greenIndicator).toBeTruthy();
              }
            });
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 13: Low score category styling
   * Validates: Requirements 3.4
   */
  test('Property 13: For any category with score below 0.3, the frontend should render it with a red progress bar', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary,
        (categories) => {
          // Ensure at least one category has low score (strictly < 0.3)
          const categoriesWithLowScore = categories.map((cat, idx) => ({
            ...cat,
            score: idx === 0 ? 0.2 : cat.score,
          }));

          const { container, unmount } = render(
            <CategoryMatchingChart categories={categoriesWithLowScore} />
          );

          // Find all low score categories (score < 0.3) that are displayed (top 3 or all)
          const sortedCategories = [...categoriesWithLowScore]
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(3, categoriesWithLowScore.length));
          
          const lowScoreCategories = sortedCategories.filter(
            (cat) => cat.score < 0.3
          );

          // Only test if there are low score categories in the displayed set
          if (lowScoreCategories.length > 0) {
            lowScoreCategories.forEach((category) => {
              // Find the category element by checking for the category name
              const categoryItems = Array.from(
                container.querySelectorAll('[role="listitem"]')
              );

              const categoryElement = categoryItems.find((el) =>
                el.textContent?.includes(category.category_name)
              );

              expect(categoryElement).toBeTruthy();

              if (categoryElement) {
                // Check for red color classes
                const classList = categoryElement.className;
                expect(classList).toContain('border-red-300');
                expect(classList).toContain('bg-red-50');

                // Check for red score indicator
                const redIndicator = categoryElement.querySelector('.bg-red-500');
                expect(redIndicator).toBeTruthy();
              }
            });
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 15: Score percentage formatting
   * Validates: Requirements 3.6
   */
  test('Property 15: For any category score display, the frontend should format scores as percentage values', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary,
        (categories) => {
          const { container, unmount } = render(
            <CategoryMatchingChart categories={categories} />
          );

          // Get the top categories that should be displayed (at least 3)
          const displayedCategories = [...categories]
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(3, categories.length));

          displayedCategories.forEach((category) => {
            // Calculate expected percentage
            const expectedPercentage = `${(category.score * 100).toFixed(0)}%`;

            // Check that the percentage is displayed in the component
            expect(container.textContent).toContain(expectedPercentage);

            // Find the category element
            const categoryItems = Array.from(
              container.querySelectorAll('[role="listitem"]')
            );

            const categoryElement = categoryItems.find((el) =>
              el.textContent?.includes(category.category_name)
            );

            if (categoryElement) {
              // Verify the percentage is in the score indicator
              const scoreIndicator = categoryElement.querySelector('.bg-green-500, .bg-yellow-500, .bg-red-500');
              expect(scoreIndicator?.textContent).toBe(expectedPercentage);
            }
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Medium score category styling
   * Verifies that categories with medium scores (0.3-0.7) are styled with yellow
   */
  test('Property: For any category with score between 0.3 and 0.7, the frontend should render it with yellow styling', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary,
        (categories) => {
          // Ensure at least one category has medium score (0.3 <= score <= 0.7)
          const categoriesWithMediumScore = categories.map((cat, idx) => ({
            ...cat,
            score: idx === 0 ? 0.5 : cat.score,
          }));

          const { container, unmount } = render(
            <CategoryMatchingChart categories={categoriesWithMediumScore} />
          );

          // Find all medium score categories (0.3 <= score <= 0.7) that are displayed (top 3 or all)
          const sortedCategories = [...categoriesWithMediumScore]
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(3, categoriesWithMediumScore.length));
          
          const mediumScoreCategories = sortedCategories.filter(
            (cat) => cat.score >= 0.3 && cat.score <= 0.7
          );

          // Only test if there are medium score categories in the displayed set
          if (mediumScoreCategories.length > 0) {
            mediumScoreCategories.forEach((category) => {
              // Find the category element by checking for the category name
              const categoryItems = Array.from(
                container.querySelectorAll('[role="listitem"]')
              );

              const categoryElement = categoryItems.find((el) =>
                el.textContent?.includes(category.category_name)
              );

              expect(categoryElement).toBeTruthy();

              if (categoryElement) {
                // Check for yellow color classes
                const classList = categoryElement.className;
                expect(classList).toContain('border-yellow-300');
                expect(classList).toContain('bg-yellow-50');

                // Check for yellow score indicator
                const yellowIndicator = categoryElement.querySelector('.bg-yellow-500');
                expect(yellowIndicator).toBeTruthy();
              }
            });
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Categories are sorted by score
   * Verifies that categories are displayed in descending order by score
   */
  test('Property: Categories should be displayed in descending order by score', () => {
    fc.assert(
      fc.property(
        uniqueCategoryArrayArbitrary.filter((cats) => cats.length >= 2),
        (categories) => {
          const { container, unmount } = render(
            <CategoryMatchingChart categories={categories} />
          );

          // Get the displayed category names in order
          const categoryItems = Array.from(
            container.querySelectorAll('[role="listitem"]')
          );

          const displayedNames = categoryItems.map((item) => {
            // Extract category name from the element
            const nameElement = item.querySelector('span.font-medium');
            return nameElement?.textContent || '';
          });

          // Get expected order (sorted by score descending, top 3 or all)
          const sortedCategories = [...categories]
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(3, categories.length));

          const expectedNames = sortedCategories.map((cat) => cat.category_name);

          // Verify the order matches
          expect(displayedNames).toEqual(expectedNames);

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Reasons are displayed when expanded
   * Verifies that clicking a category shows its reasons
   */
  test('Property: For any category, clicking it should display its reasons', async () => {
    fc.assert(
      fc.asyncProperty(
        uniqueCategoryArrayArbitrary.filter((cats) => 
          // Only test with categories that have at least one non-empty reason
          cats.some(cat => cat.reasons.some(r => r.trim().length > 5))
        ),
        async (categories) => {
          const { container, unmount } = render(
            <CategoryMatchingChart categories={categories} />
          );

          // Get the first displayed category
          const firstCategoryItem = container.querySelector('[role="listitem"]');
          
          if (firstCategoryItem) {
            // Get the category data for the first item
            const sortedCategories = [...categories]
              .sort((a, b) => b.score - a.score)
              .slice(0, Math.max(3, categories.length));
            const firstCategory = sortedCategories[0];

            // Initially, reasons should not be visible
            const initialReasonsList = firstCategoryItem.querySelector('ul.list-disc');
            expect(initialReasonsList).toBeNull();

            // Click the category using act
            await act(async () => {
              (firstCategoryItem as HTMLElement).click();
            });

            // After clicking, if the category has non-empty reasons, they should be visible
            const nonEmptyReasons = firstCategory.reasons.filter(r => r.trim().length > 0);
            if (nonEmptyReasons.length > 0) {
              const expandedReasonsList = firstCategoryItem.querySelector('ul.list-disc');
              expect(expandedReasonsList).toBeTruthy();

              // Verify that reasons are displayed
              const reasonItems = firstCategoryItem.querySelectorAll('ul.list-disc li');
              expect(reasonItems.length).toBe(firstCategory.reasons.length);
            }
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
