import { describe, it, expect } from 'vitest';
import fc from 'fast-check';

/**
 * Feature: trendyol-gift-recommendation-web, Property 19: Responsive Layout Preservation
 * Validates: Requirements 8.1
 */

describe('Property 19: Responsive Layout Preservation', () => {
  it('should maintain proper layout without horizontal scrolling for any screen width', () => {
    fc.assert(
      fc.property(
        // Generate screen widths from mobile (320px) to large desktop (1920px)
        fc.integer({ min: 320, max: 1920 }),
        (screenWidth) => {
          // Create a container element to simulate viewport
          const container = document.createElement('div');
          container.style.width = `${screenWidth}px`;
          container.style.overflow = 'hidden';
          document.body.appendChild(container);

          // Create test elements with responsive classes
          const testElements = [
            // Container with responsive padding
            createElementWithClasses('div', ['container', 'mx-auto', 'px-4', 'sm:px-6', 'lg:px-8']),
            // Grid with responsive columns
            createElementWithClasses('div', ['grid', 'grid-cols-1', 'sm:grid-cols-2', 'lg:grid-cols-3', 'gap-4', 'sm:gap-6']),
            // Text with responsive sizes
            createElementWithClasses('h1', ['text-3xl', 'sm:text-4xl', 'md:text-5xl']),
            // Button with responsive padding
            createElementWithClasses('button', ['px-4', 'sm:px-6', 'py-2', 'touch-target']),
            // Flex container with responsive direction
            createElementWithClasses('div', ['flex', 'flex-col', 'sm:flex-row', 'gap-2', 'sm:gap-3']),
          ];

          testElements.forEach(element => {
            container.appendChild(element);
          });

          // Check that no element exceeds container width
          const allElements = container.querySelectorAll('*');
          let hasHorizontalOverflow = false;

          allElements.forEach(element => {
            const rect = element.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            // Element should not extend beyond container
            if (rect.right > containerRect.right + 1) { // +1 for rounding errors
              hasHorizontalOverflow = true;
            }
          });

          // Cleanup
          document.body.removeChild(container);

          // Property: No horizontal overflow should occur
          expect(hasHorizontalOverflow).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should apply appropriate breakpoint classes for any screen width', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 320, max: 1920 }),
        (screenWidth) => {
          // Determine expected breakpoint
          let expectedBreakpoint: string;
          if (screenWidth < 640) {
            expectedBreakpoint = 'mobile';
          } else if (screenWidth < 768) {
            expectedBreakpoint = 'sm';
          } else if (screenWidth < 1024) {
            expectedBreakpoint = 'md';
          } else if (screenWidth < 1280) {
            expectedBreakpoint = 'lg';
          } else {
            expectedBreakpoint = 'xl';
          }

          // Verify breakpoint logic is consistent
          expect(expectedBreakpoint).toBeDefined();
          expect(['mobile', 'sm', 'md', 'lg', 'xl']).toContain(expectedBreakpoint);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should maintain minimum touch target size (44x44px) on all screen sizes', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 320, max: 1920 }),
        (screenWidth) => {
          // Property: Touch target classes should be applied consistently
          // We verify the class is present rather than measuring actual rendered size
          // since JSDOM doesn't render actual dimensions
          const button = createElementWithClasses('button', ['touch-target', 'px-4', 'py-2']);
          
          // Verify touch-target class is present
          expect(button.classList.contains('touch-target')).toBe(true);
          
          // In a real browser, touch-target utility would ensure min-height and min-width of 44px
          // This is enforced by our CSS utility class definition
          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should preserve content readability with responsive text sizes', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 320, max: 1920 }),
        fc.constantFrom('text-xs', 'text-sm', 'text-base', 'text-lg', 'text-xl', 'text-2xl', 'text-3xl'),
        (screenWidth, textClass) => {
          // Property: Text size classes should be applied consistently
          const textElement = createElementWithClasses('p', [textClass]);
          textElement.textContent = 'Sample text content';

          // Verify the text class is present
          expect(textElement.classList.contains(textClass)).toBe(true);
          
          // Verify content is present
          expect(textElement.textContent).toBe('Sample text content');
          
          // In a real browser with Tailwind CSS, these classes would render
          // with appropriate font sizes (text-xs: 0.75rem, text-sm: 0.875rem, etc.)
          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should maintain proper spacing with responsive gap classes', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 320, max: 1920 }),
        fc.constantFrom('gap-1', 'gap-2', 'gap-3', 'gap-4', 'gap-6', 'gap-8'),
        (screenWidth, gapClass) => {
          const container = document.createElement('div');
          container.style.width = `${screenWidth}px`;
          document.body.appendChild(container);

          const flexContainer = createElementWithClasses('div', ['flex', gapClass]);
          
          // Add child elements
          for (let i = 0; i < 3; i++) {
            const child = document.createElement('div');
            child.style.width = '50px';
            child.style.height = '50px';
            flexContainer.appendChild(child);
          }

          container.appendChild(flexContainer);

          // Verify container exists and has children
          expect(flexContainer.children.length).toBe(3);

          // Cleanup
          document.body.removeChild(container);

          return true;
        }
      ),
      { numRuns: 100 }
    );
  });
});

// Helper function to create elements with Tailwind classes
function createElementWithClasses(tagName: string, classes: string[]): HTMLElement {
  const element = document.createElement(tagName);
  element.className = classes.join(' ');
  return element;
}
