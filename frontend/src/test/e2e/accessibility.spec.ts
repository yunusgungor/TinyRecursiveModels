import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

/**
 * E2E Test: Accessibility Flow
 * 
 * This test validates accessibility compliance:
 * 1. ARIA labels and roles
 * 2. Keyboard navigation
 * 3. Screen reader compatibility
 * 4. Focus management
 * 5. Color contrast
 * 6. Color-blind friendly design
 */

test.describe('Accessibility Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/recommendations');
    await page.waitForLoadState('networkidle');
  });

  test('should pass automated accessibility checks', async ({ page }) => {
    // Run axe accessibility tests on the page
    const accessibilityScanResults = await new AxeBuilder({ page })
      .analyze();
    
    // Verify no violations
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should have proper ARIA labels on all interactive elements', async ({ page }) => {
    // Check gift card has proper ARIA labels
    const giftCard = page.locator('[data-testid="gift-card"]').first();
    await expect(giftCard).toHaveAttribute('role', 'article');
    
    // Check show details button has ARIA label
    const showDetailsButton = page.locator('[data-testid="show-details-button"]').first();
    const ariaLabel = await showDetailsButton.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
    expect(ariaLabel).toContain('Detaylı');
    
    // Check confidence indicator has ARIA label
    const confidenceIndicator = page.locator('[data-testid="confidence-indicator"]').first();
    const confidenceAriaLabel = await confidenceIndicator.getAttribute('aria-label');
    expect(confidenceAriaLabel).toBeTruthy();
    
    // Open reasoning panel
    await showDetailsButton.click();
    
    // Check reasoning panel has proper ARIA attributes
    const reasoningPanel = page.locator('[data-testid="reasoning-panel"]');
    await expect(reasoningPanel).toHaveAttribute('role', 'dialog');
    await expect(reasoningPanel).toHaveAttribute('aria-modal', 'true');
    
    // Check close button has ARIA label
    const closeButton = page.locator('[data-testid="close-panel-button"]');
    const closeAriaLabel = await closeButton.getAttribute('aria-label');
    expect(closeAriaLabel).toBeTruthy();
  });

  test('should support complete keyboard navigation', async ({ page }) => {
    // Start from first gift card
    await page.keyboard.press('Tab');
    
    // Verify focus is on first interactive element
    let focusedElement = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
    expect(focusedElement).toBeTruthy();
    
    // Tab through gift cards
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press('Tab');
    }
    
    // Press Enter on show details button
    await page.keyboard.press('Enter');
    
    // Verify reasoning panel opens
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
    
    // Tab through reasoning panel elements
    await page.keyboard.press('Tab');
    focusedElement = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
    expect(focusedElement).toBeTruthy();
    
    // Press Escape to close panel
    await page.keyboard.press('Escape');
    
    // Verify panel closes
    await expect(page.locator('[data-testid="reasoning-panel"]')).not.toBeVisible();
  });

  test('should trap focus within modal dialogs', async ({ page }) => {
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
    
    // Click confidence indicator to open modal
    await page.locator('[data-testid="confidence-indicator"]').click();
    await expect(page.locator('[data-testid="confidence-modal"]')).toBeVisible();
    
    // Get all focusable elements in modal
    const modal = page.locator('[data-testid="confidence-modal"]');
    const focusableElements = await modal.locator('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])').all();
    
    expect(focusableElements.length).toBeGreaterThan(0);
    
    // Tab through all elements
    for (let i = 0; i < focusableElements.length + 1; i++) {
      await page.keyboard.press('Tab');
    }
    
    // Focus should cycle back to first element in modal
    const focusedElement = await page.evaluate(() => {
      const activeEl = document.activeElement;
      return activeEl?.closest('[data-testid="confidence-modal"]') !== null;
    });
    
    expect(focusedElement).toBeTruthy();
  });

  test('should provide screen reader announcements', async ({ page }) => {
    // Check for live regions
    const liveRegion = page.locator('[aria-live="polite"]');
    
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    
    // Verify announcement region exists
    await expect(liveRegion).toBeAttached();
    
    // Check that status messages are announced
    const statusMessage = await liveRegion.textContent();
    expect(statusMessage).toBeTruthy();
  });

  test('should have proper heading hierarchy', async ({ page }) => {
    // Check main heading
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
    
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    
    // Check section headings
    const h2Elements = page.locator('[data-testid="reasoning-panel"] h2');
    const h2Count = await h2Elements.count();
    expect(h2Count).toBeGreaterThan(0);
    
    // Verify no heading levels are skipped
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
    const headingLevels = await Promise.all(
      headings.map(h => h.evaluate(el => parseInt(el.tagName.substring(1))))
    );
    
    // Check that heading levels don't skip (e.g., h1 -> h3)
    for (let i = 1; i < headingLevels.length; i++) {
      const diff = headingLevels[i] - headingLevels[i - 1];
      expect(diff).toBeLessThanOrEqual(1);
    }
  });

  test('should have sufficient color contrast', async ({ page }) => {
    // Run axe color contrast check
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .analyze();
    
    // Filter for color contrast violations
    const contrastViolations = accessibilityScanResults.violations.filter(
      v => v.id === 'color-contrast'
    );
    
    expect(contrastViolations).toHaveLength(0);
  });

  test('should provide color-blind friendly visual cues', async ({ page }) => {
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    
    // Check tool selection uses icons in addition to colors
    const selectedTool = page.locator('[data-testid="tool-card"][data-selected="true"]').first();
    await expect(selectedTool).toBeVisible();
    
    // Verify checkmark icon is present
    const checkmarkIcon = selectedTool.locator('[data-testid="checkmark-icon"]');
    await expect(checkmarkIcon).toBeVisible();
    
    // Check category bars use patterns or labels
    const categoryBar = page.locator('[data-testid="category-bar"]').first();
    await expect(categoryBar).toBeVisible();
    
    // Verify bar has text label
    const barLabel = await categoryBar.textContent();
    expect(barLabel).toBeTruthy();
    expect(barLabel?.length).toBeGreaterThan(0);
  });

  test('should support screen reader navigation of charts', async ({ page }) => {
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    
    // Check category chart has proper ARIA labels
    const categoryChart = page.locator('[data-testid="category-matching-chart"]');
    await expect(categoryChart).toHaveAttribute('role', 'img');
    
    const chartAriaLabel = await categoryChart.getAttribute('aria-label');
    expect(chartAriaLabel).toBeTruthy();
    expect(chartAriaLabel).toContain('kategori');
    
    // Check chart has accessible description
    const chartDescription = await categoryChart.getAttribute('aria-describedby');
    if (chartDescription) {
      const descriptionElement = page.locator(`#${chartDescription}`);
      await expect(descriptionElement).toBeAttached();
    }
    
    // Check attention weights chart
    const attentionChart = page.locator('[data-testid="attention-weights-chart"]');
    await expect(attentionChart).toHaveAttribute('role', 'img');
    
    const attentionAriaLabel = await attentionChart.getAttribute('aria-label');
    expect(attentionAriaLabel).toBeTruthy();
  });

  test('should provide alternative text for images', async ({ page }) => {
    // Check gift images have alt text
    const giftImage = page.locator('[data-testid="gift-card"]').first().locator('img');
    
    if (await giftImage.count() > 0) {
      const altText = await giftImage.getAttribute('alt');
      expect(altText).toBeTruthy();
      expect(altText?.length).toBeGreaterThan(0);
    }
  });

  test('should handle focus management when opening/closing panels', async ({ page }) => {
    // Get initial focused element
    const showDetailsButton = page.locator('[data-testid="show-details-button"]').first();
    await showDetailsButton.focus();
    
    // Open panel
    await showDetailsButton.click();
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
    
    // Focus should move into panel
    await page.waitForTimeout(100);
    const focusedInPanel = await page.evaluate(() => {
      const activeEl = document.activeElement;
      return activeEl?.closest('[data-testid="reasoning-panel"]') !== null;
    });
    expect(focusedInPanel).toBeTruthy();
    
    // Close panel
    await page.keyboard.press('Escape');
    await expect(page.locator('[data-testid="reasoning-panel"]')).not.toBeVisible();
    
    // Focus should return to trigger button
    await page.waitForTimeout(100);
    const focusedElement = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
    expect(focusedElement).toBe('show-details-button');
  });

  test('should support reduced motion preferences', async ({ page, context }) => {
    // Set prefers-reduced-motion
    await context.addInitScript(() => {
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: (query: string) => ({
          matches: query === '(prefers-reduced-motion: reduce)',
          media: query,
          onchange: null,
          addListener: () => {},
          removeListener: () => {},
          addEventListener: () => {},
          removeEventListener: () => {},
          dispatchEvent: () => true,
        }),
      });
    });
    
    await page.goto('/recommendations');
    await page.waitForLoadState('networkidle');
    
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    
    // Verify panel opens without animation (instant)
    const panel = page.locator('[data-testid="reasoning-panel"]');
    await expect(panel).toBeVisible();
    
    // Check that transitions are disabled
    const transitionDuration = await panel.evaluate(el => 
      window.getComputedStyle(el).transitionDuration
    );
    
    // Should be 0s or very short
    expect(transitionDuration === '0s' || transitionDuration === '0.01s').toBeTruthy();
  });

  test('should provide skip links for keyboard users', async ({ page }) => {
    // Tab to first element
    await page.keyboard.press('Tab');
    
    // Check if skip link is visible
    const skipLink = page.locator('[data-testid="skip-to-content"]');
    
    if (await skipLink.count() > 0) {
      await expect(skipLink).toBeFocused();
      
      // Activate skip link
      await page.keyboard.press('Enter');
      
      // Verify focus moved to main content
      const mainContent = page.locator('main');
      const mainIsFocused = await mainContent.evaluate(el => 
        document.activeElement === el || el.contains(document.activeElement)
      );
      expect(mainIsFocused).toBeTruthy();
    }
  });
});
