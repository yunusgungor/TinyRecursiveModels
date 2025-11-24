import { describe, test, expect } from 'vitest';
import { render } from '@testing-library/react';
import fc from 'fast-check';
import { ToolSelectionCard } from '../ToolSelectionCard';
import { ToolSelectionReasoning } from '@/types/reasoning';

/**
 * Property-based tests for ToolSelectionCard component
 */

// Arbitrary generator for tool names
const toolNameArbitrary = fc.constantFrom(
  'review_analysis',
  'trend_analysis',
  'inventory_check',
  'price_comparison',
  'category_filter'
);

// Arbitrary generator for ToolSelectionReasoning
const toolSelectionReasoningArbitrary = fc.record({
  name: toolNameArbitrary,
  selected: fc.boolean(),
  score: fc.float({ min: 0, max: 1, noDefaultInfinity: true, noNaN: true }),
  reason: fc.string({ minLength: 10, maxLength: 100 }),
  confidence: fc.float({ min: 0, max: 1, noDefaultInfinity: true, noNaN: true }),
  priority: fc.integer({ min: 1, max: 10 }),
  factors: fc.option(
    fc.dictionary(
      fc.string({ minLength: 3, maxLength: 20 }),
      fc.float({ min: 0, max: 1, noDefaultInfinity: true, noNaN: true })
    ),
    { nil: undefined }
  ),
}) as fc.Arbitrary<ToolSelectionReasoning>;

// Generator for unique tool arrays (no duplicate names)
const uniqueToolArrayArbitrary = fc
  .array(toolSelectionReasoningArbitrary, { minLength: 1, maxLength: 5 })
  .map((tools) => {
    // Remove duplicates by name, keeping first occurrence
    const seen = new Set<string>();
    return tools.filter((tool) => {
      if (seen.has(tool.name)) {
        return false;
      }
      seen.add(tool.name);
      return true;
    });
  })
  .filter((tools) => tools.length > 0); // Ensure at least one tool remains

describe('ToolSelectionCard Property Tests', () => {
  /**
   * Feature: frontend-reasoning-visualization, Property 7: Selected tool styling
   * Validates: Requirements 2.3
   */
  test('Property 7: For any selected tool, the frontend should render it with green color and checkmark icon', () => {
    fc.assert(
      fc.property(
        uniqueToolArrayArbitrary,
        (tools) => {
          // Ensure at least one tool is selected
          const toolsWithSelection = tools.map((tool, idx) => ({
            ...tool,
            selected: idx === 0 ? true : tool.selected,
          }));

          const { container, unmount } = render(
            <ToolSelectionCard toolSelection={toolsWithSelection} />
          );

          // Find all selected tools
          const selectedTools = toolsWithSelection.filter((t) => t.selected);

          selectedTools.forEach((tool) => {
            // Find the tool element by checking for the tool name in the content
            const toolElements = Array.from(
              container.querySelectorAll('[role="listitem"]')
            );

            const toolElement = toolElements.find((el) =>
              el.textContent?.includes(
                tool.name === 'review_analysis'
                  ? 'Yorum Analizi'
                  : tool.name === 'trend_analysis'
                  ? 'Trend Analizi'
                  : tool.name === 'inventory_check'
                  ? 'Stok Kontrolü'
                  : tool.name === 'price_comparison'
                  ? 'Fiyat Karşılaştırma'
                  : tool.name === 'category_filter'
                  ? 'Kategori Filtresi'
                  : tool.name
              )
            );

            expect(toolElement).toBeTruthy();

            if (toolElement) {
              // Check for green color classes
              const classList = toolElement.className;
              expect(classList).toContain('border-green-300');
              expect(classList).toContain('bg-green-50');

              // Check for checkmark icon (SVG path with specific d attribute)
              const checkmarkPath = toolElement.querySelector(
                'path[d="M5 13l4 4L19 7"]'
              );
              expect(checkmarkPath).toBeTruthy();

              // Check for green icon background
              const iconContainer = toolElement.querySelector('.bg-green-500');
              expect(iconContainer).toBeTruthy();
            }
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 8: Unselected tool styling
   * Validates: Requirements 2.4
   */
  test('Property 8: For any unselected tool, the frontend should render it with gray color', () => {
    fc.assert(
      fc.property(
        uniqueToolArrayArbitrary,
        (tools) => {
          // Ensure at least one tool is unselected
          const toolsWithUnselection = tools.map((tool, idx) => ({
            ...tool,
            selected: idx === 0 ? false : tool.selected,
          }));

          const { container, unmount } = render(
            <ToolSelectionCard toolSelection={toolsWithUnselection} />
          );

          // Find all unselected tools
          const unselectedTools = toolsWithUnselection.filter((t) => !t.selected);

          unselectedTools.forEach((tool) => {
            // Find the tool element by checking for the tool name in the content
            const toolElements = Array.from(
              container.querySelectorAll('[role="listitem"]')
            );

            const toolElement = toolElements.find((el) =>
              el.textContent?.includes(
                tool.name === 'review_analysis'
                  ? 'Yorum Analizi'
                  : tool.name === 'trend_analysis'
                  ? 'Trend Analizi'
                  : tool.name === 'inventory_check'
                  ? 'Stok Kontrolü'
                  : tool.name === 'price_comparison'
                  ? 'Fiyat Karşılaştırma'
                  : tool.name === 'category_filter'
                  ? 'Kategori Filtresi'
                  : tool.name
              )
            );

            expect(toolElement).toBeTruthy();

            if (toolElement) {
              // Check for gray color classes
              const classList = toolElement.className;
              expect(classList).toContain('border-gray-200');
              expect(classList).toContain('bg-gray-50');

              // Check for X icon (cross) instead of checkmark
              const crossPath = toolElement.querySelector(
                'path[d="M6 18L18 6M6 6l12 12"]'
              );
              expect(crossPath).toBeTruthy();

              // Check for gray icon background
              const iconContainer = toolElement.querySelector('.bg-gray-300');
              expect(iconContainer).toBeTruthy();
            }
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: frontend-reasoning-visualization, Property 9: Low confidence tooltip
   * Validates: Requirements 2.5
   */
  test('Property 9: For any tool with confidence below 0.5, the frontend should display a tooltip explaining the low confidence reason', () => {
    fc.assert(
      fc.property(
        uniqueToolArrayArbitrary,
        (tools) => {
          // Ensure at least one tool has low confidence (strictly less than 0.5)
          const toolsWithLowConfidence = tools.map((tool, idx) => ({
            ...tool,
            confidence: idx === 0 ? 0.3 : tool.confidence,
          }));

          const { container, unmount } = render(
            <ToolSelectionCard toolSelection={toolsWithLowConfidence} />
          );

          // Find all low confidence tools (strictly less than 0.5)
          const lowConfidenceTools = toolsWithLowConfidence.filter(
            (t) => t.confidence < 0.5
          );

          lowConfidenceTools.forEach((tool) => {
            // Find the tool element by checking for the tool name in the content
            const toolElements = Array.from(
              container.querySelectorAll('[role="listitem"]')
            );

            const toolElement = toolElements.find((el) =>
              el.textContent?.includes(
                tool.name === 'review_analysis'
                  ? 'Yorum Analizi'
                  : tool.name === 'trend_analysis'
                  ? 'Trend Analizi'
                  : tool.name === 'inventory_check'
                  ? 'Stok Kontrolü'
                  : tool.name === 'price_comparison'
                  ? 'Fiyat Karşılaştırma'
                  : tool.name === 'category_filter'
                  ? 'Kategori Filtresi'
                  : tool.name
              )
            );

            expect(toolElement).toBeTruthy();

            if (toolElement) {
              // Check for warning icon (yellow background)
              const warningIcon = toolElement.querySelector('.bg-yellow-100');
              expect(warningIcon).toBeTruthy();

              // Check for warning icon SVG
              const warningPath = toolElement.querySelector(
                'path[fill-rule="evenodd"]'
              );
              expect(warningPath).toBeTruthy();

              // Check for aria-label indicating low confidence warning
              const warningElement = toolElement.querySelector(
                '[aria-label="Düşük güven uyarısı"]'
              );
              expect(warningElement).toBeTruthy();
            }
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Tool information completeness
   * Verifies that all required tool information is displayed
   */
  test('Property: For any tool, the frontend should display selection status, confidence score, and priority', () => {
    fc.assert(
      fc.property(
        uniqueToolArrayArbitrary,
        (tools) => {
          const { container, unmount } = render(
            <ToolSelectionCard toolSelection={tools} />
          );

          tools.forEach((tool) => {
            // Check that confidence percentage is displayed
            const confidenceText = `${(tool.confidence * 100).toFixed(0)}%`;
            expect(container.textContent).toContain(confidenceText);

            // Check that priority is displayed
            const priorityText = `Öncelik: ${tool.priority}`;
            expect(container.textContent).toContain(priorityText);

            // Check that tool name is displayed (in Turkish)
            const displayName =
              tool.name === 'review_analysis'
                ? 'Yorum Analizi'
                : tool.name === 'trend_analysis'
                ? 'Trend Analizi'
                : tool.name === 'inventory_check'
                ? 'Stok Kontrolü'
                : tool.name === 'price_comparison'
                ? 'Fiyat Karşılaştırma'
                : tool.name === 'category_filter'
                ? 'Kategori Filtresi'
                : tool.name;
            expect(container.textContent).toContain(displayName);
          });

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
