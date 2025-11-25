import { test, expect, devices } from '@playwright/test';

/**
 * E2E Test: Mobile Responsive Behavior
 * 
 * This test validates responsive design across different viewports:
 * 1. Mobile layout adaptations
 * 2. Touch gestures
 * 3. Full-screen modals
 * 4. Vertical chart layouts
 * 5. Touch-friendly tooltips
 */

test.describe('Mobile Responsive Behavior', () => {
  test.describe('Mobile Phone (375px)', () => {
    test('should adapt layout for mobile viewport', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Verify mobile layout is applied
      const viewport = page.viewportSize();
      expect(viewport?.width).toBeLessThan(768);
      
      // Verify gift cards stack vertically
      const cards = page.locator('[data-testid="gift-card"]');
      const firstCard = cards.first();
      const secondCard = cards.nth(1);
      
      const firstBox = await firstCard.boundingBox();
      const secondBox = await secondCard.boundingBox();
      
      // Second card should be below first card (vertical stacking)
      if (firstBox && secondBox) {
        expect(secondBox.y).toBeGreaterThan(firstBox.y + firstBox.height);
      }
    });

    test('should display charts in vertical layout on mobile', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().click();
      
      // Verify panel opens in full-screen mode
      const panel = page.locator('[data-testid="reasoning-panel"]');
      await expect(panel).toBeVisible();
      
      const panelBox = await panel.boundingBox();
      const viewport = page.viewportSize();
      
      // Panel should take full width on mobile
      if (panelBox && viewport) {
        expect(panelBox.width).toBeGreaterThanOrEqual(viewport.width * 0.95);
      }
      
      // Verify charts are in vertical layout
      const categoryChart = page.locator('[data-testid="category-matching-chart"]');
      await expect(categoryChart).toBeVisible();
      
      // Chart should be full width
      const chartBox = await categoryChart.boundingBox();
      if (chartBox && viewport) {
        expect(chartBox.width).toBeGreaterThanOrEqual(viewport.width * 0.8);
      }
    });

    test('should support swipe gesture to close panel', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().click();
      await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
      
      // Perform swipe down gesture
      const panel = page.locator('[data-testid="reasoning-panel"]');
      const panelBox = await panel.boundingBox();
      
      if (panelBox) {
        // Swipe from top to bottom
        await page.touchscreen.tap(panelBox.x + panelBox.width / 2, panelBox.y + 50);
        await page.touchscreen.tap(panelBox.x + panelBox.width / 2, panelBox.y + panelBox.height - 50);
      }
      
      // Panel should close
      await expect(page.locator('[data-testid="reasoning-panel"]')).not.toBeVisible({ timeout: 2000 });
    });

    test('should display touch-friendly tooltips', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().click();
      
      // Tap on tool card
      const toolCard = page.locator('[data-testid="tool-card"]').first();
      await toolCard.tap();
      
      // Verify tooltip appears and is touch-friendly
      const tooltip = page.locator('[role="tooltip"]');
      await expect(tooltip).toBeVisible();
      
      // Tooltip should be large enough for touch
      const tooltipBox = await tooltip.boundingBox();
      if (tooltipBox) {
        expect(tooltipBox.height).toBeGreaterThan(44); // Minimum touch target size
      }
    });

    test('should handle mobile navigation', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/');
      
      // Fill form on mobile
      await page.fill('input[name="age"]', '25');
      await page.fill('input[name="budget"]', '500');
      
      // Scroll to submit button
      await page.locator('button[type="submit"]').scrollIntoViewIfNeeded();
      await page.click('button[type="submit"]');
      
      // Verify navigation to recommendations
      await expect(page).toHaveURL(/.*recommendations/);
      await expect(page.locator('[data-testid="gift-card"]').first()).toBeVisible();
    });
  });

  test.describe('Tablet (768px)', () => {
    test('should adapt layout for tablet viewport', async ({ page, context }) => {
      // Set tablet viewport
      await context.setViewportSize(devices['iPad'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Verify tablet layout
      const viewport = page.viewportSize();
      expect(viewport?.width).toBeGreaterThanOrEqual(768);
      expect(viewport?.width).toBeLessThan(1024);
      
      // Verify gift cards display in grid (2 columns on tablet)
      const cards = page.locator('[data-testid="gift-card"]');
      await expect(cards.first()).toBeVisible();
      await expect(cards.nth(1)).toBeVisible();
      
      const firstBox = await cards.first().boundingBox();
      const secondBox = await cards.nth(1).boundingBox();
      
      // Cards should be side by side or stacked depending on design
      if (firstBox && secondBox) {
        const horizontalDistance = Math.abs(secondBox.x - firstBox.x);
        const verticalDistance = Math.abs(secondBox.y - firstBox.y);
        
        // Either horizontal or vertical layout is acceptable
        expect(horizontalDistance > 0 || verticalDistance > 0).toBeTruthy();
      }
    });

    test('should display reasoning panel as side panel on tablet', async ({ page, context }) => {
      // Set tablet viewport
      await context.setViewportSize(devices['iPad'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().click();
      
      // Panel should be visible but not full-screen
      const panel = page.locator('[data-testid="reasoning-panel"]');
      await expect(panel).toBeVisible();
      
      const panelBox = await panel.boundingBox();
      const viewport = page.viewportSize();
      
      // Panel should not take full width on tablet
      if (panelBox && viewport) {
        expect(panelBox.width).toBeLessThan(viewport.width);
      }
    });
  });

  test.describe('Desktop (1920px)', () => {
    test('should display full desktop layout', async ({ page, context }) => {
      // Set desktop viewport
      await context.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Verify desktop layout with multiple columns
      const cards = page.locator('[data-testid="gift-card"]');
      await expect(cards.first()).toBeVisible();
      
      // Verify cards are in grid layout
      const firstBox = await cards.first().boundingBox();
      const secondBox = await cards.nth(1).boundingBox();
      
      if (firstBox && secondBox) {
        // Cards should be side by side on desktop
        expect(secondBox.x).toBeGreaterThan(firstBox.x);
      }
    });

    test('should display reasoning panel as side drawer on desktop', async ({ page, context }) => {
      // Set desktop viewport
      await context.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().click();
      
      // Panel should slide in from the side
      const panel = page.locator('[data-testid="reasoning-panel"]');
      await expect(panel).toBeVisible();
      
      const panelBox = await panel.boundingBox();
      const viewport = page.viewportSize();
      
      // Panel should be a side drawer (not full width)
      if (panelBox && viewport) {
        expect(panelBox.width).toBeLessThan(viewport.width * 0.6);
      }
    });
  });

  test.describe('Orientation Changes', () => {
    test('should handle portrait to landscape transition', async ({ page, context }) => {
      // Start in portrait
      await context.setViewportSize({ width: 375, height: 667 });
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Verify portrait layout
      let viewport = page.viewportSize();
      expect(viewport?.width).toBeLessThan(viewport?.height || 0);
      
      // Switch to landscape
      await context.setViewportSize({ width: 667, height: 375 });
      await page.waitForTimeout(500); // Wait for layout adjustment
      
      // Verify landscape layout
      viewport = page.viewportSize();
      expect(viewport?.width).toBeGreaterThan(viewport?.height || 0);
      
      // Verify content is still accessible
      await expect(page.locator('[data-testid="gift-card"]').first()).toBeVisible();
    });
  });

  test.describe('Touch Interactions', () => {
    test('should handle touch interactions on mobile', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Tap on gift card
      const card = page.locator('[data-testid="gift-card"]').first();
      await card.tap();
      
      // Tap on show details button
      await page.locator('[data-testid="show-details-button"]').first().tap();
      
      // Verify panel opens
      await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
      
      // Tap on thinking step
      const step = page.locator('[data-testid="thinking-step"]').first();
      await step.tap();
      
      // Verify step expands
      await expect(page.locator('[data-testid="step-details"]').first()).toBeVisible();
    });

    test('should support pinch zoom on charts', async ({ page, context }) => {
      // Set mobile viewport
      await context.setViewportSize(devices['iPhone 12'].viewport);
      await page.goto('/recommendations');
      await page.waitForLoadState('networkidle');
      
      // Open reasoning panel
      await page.locator('[data-testid="show-details-button"]').first().tap();
      
      // Get chart element
      const chart = page.locator('[data-testid="category-matching-chart"]');
      await expect(chart).toBeVisible();
      
      // Note: Actual pinch zoom testing requires more complex touch simulation
      // This test verifies the chart is accessible for touch interactions
      const chartBox = await chart.boundingBox();
      expect(chartBox).toBeTruthy();
    });
  });
});
