import { test, expect } from '@playwright/test';

/**
 * E2E Test: Comparison Mode
 * 
 * This test validates the comparison functionality:
 * 1. Selecting multiple gifts
 * 2. Activating comparison mode
 * 3. Viewing side-by-side reasoning
 * 4. Comparing charts
 * 5. Exiting comparison mode
 */

test.describe('Comparison Mode', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to recommendations page
    await page.goto('/recommendations');
    await page.waitForLoadState('networkidle');
  });

  test('should enable comparison mode when multiple gifts selected', async ({ page }) => {
    // Wait for gift cards to load
    await expect(page.locator('[data-testid="gift-card"]').first()).toBeVisible();
    
    // Select first gift
    const firstCard = page.locator('[data-testid="gift-card"]').first();
    await firstCard.locator('[data-testid="select-checkbox"]').check();
    
    // Verify compare button is not visible yet (need at least 2)
    await expect(page.locator('[data-testid="compare-button"]')).not.toBeVisible();
    
    // Select second gift
    const secondCard = page.locator('[data-testid="gift-card"]').nth(1);
    await secondCard.locator('[data-testid="select-checkbox"]').check();
    
    // Verify compare button appears
    await expect(page.locator('[data-testid="compare-button"]')).toBeVisible();
    
    // Click compare button
    await page.click('[data-testid="compare-button"]');
    
    // Verify comparison view opens
    await expect(page.locator('[data-testid="comparison-view"]')).toBeVisible();
  });

  test('should display side-by-side gift comparison', async ({ page }) => {
    // Select two gifts
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    
    // Enter comparison mode
    await page.click('[data-testid="compare-button"]');
    
    // Verify both gifts are displayed side by side
    const comparisonCards = page.locator('[data-testid="comparison-card"]');
    await expect(comparisonCards).toHaveCount(2);
    
    // Verify gift information is displayed
    await expect(comparisonCards.first().locator('[data-testid="gift-name"]')).toBeVisible();
    await expect(comparisonCards.nth(1).locator('[data-testid="gift-name"]')).toBeVisible();
    
    // Verify reasoning is displayed for both
    await expect(comparisonCards.first().locator('[data-testid="reasoning-text"]')).toBeVisible();
    await expect(comparisonCards.nth(1).locator('[data-testid="reasoning-text"]')).toBeVisible();
    
    // Verify confidence indicators are displayed
    await expect(comparisonCards.first().locator('[data-testid="confidence-indicator"]')).toBeVisible();
    await expect(comparisonCards.nth(1).locator('[data-testid="confidence-indicator"]')).toBeVisible();
  });

  test('should compare category scores in same chart', async ({ page }) => {
    // Select two gifts and enter comparison mode
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    await page.click('[data-testid="compare-button"]');
    
    // Verify comparison chart is visible
    await expect(page.locator('[data-testid="comparison-category-chart"]')).toBeVisible();
    
    // Verify chart contains data for both gifts
    const chartBars = page.locator('[data-testid="comparison-category-chart"] [data-testid="category-bar"]');
    await expect(chartBars.first()).toBeVisible();
    
    // Verify different colors are used for different gifts
    const firstBar = chartBars.first();
    const firstBarColor = await firstBar.evaluate(el => window.getComputedStyle(el).fill);
    
    const secondBar = chartBars.nth(1);
    const secondBarColor = await secondBar.evaluate(el => window.getComputedStyle(el).fill);
    
    // Colors should be different
    expect(firstBarColor).not.toBe(secondBarColor);
  });

  test('should compare attention weights with overlay chart', async ({ page }) => {
    // Select two gifts and enter comparison mode
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    await page.click('[data-testid="compare-button"]');
    
    // Navigate to attention weights section
    await page.click('[data-testid="attention-weights-tab"]');
    
    // Verify overlay chart is visible
    await expect(page.locator('[data-testid="comparison-attention-chart"]')).toBeVisible();
    
    // Verify chart shows data for both gifts
    const chartElements = page.locator('[data-testid="comparison-attention-chart"] [data-testid="attention-bar"]');
    await expect(chartElements.first()).toBeVisible();
    
    // Hover over bar to see tooltip
    await chartElements.first().hover();
    await expect(page.locator('[role="tooltip"]')).toBeVisible();
  });

  test('should exit comparison mode and return to normal view', async ({ page }) => {
    // Select two gifts and enter comparison mode
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    await page.click('[data-testid="compare-button"]');
    
    // Verify comparison view is visible
    await expect(page.locator('[data-testid="comparison-view"]')).toBeVisible();
    
    // Click exit comparison button
    await page.click('[data-testid="exit-comparison-button"]');
    
    // Verify comparison view is closed
    await expect(page.locator('[data-testid="comparison-view"]')).not.toBeVisible();
    
    // Verify normal gift cards are visible again
    await expect(page.locator('[data-testid="gift-card"]').first()).toBeVisible();
    
    // Verify selections are cleared
    const firstCheckbox = page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]');
    await expect(firstCheckbox).not.toBeChecked();
  });

  test('should handle comparison with three gifts', async ({ page }) => {
    // Select three gifts
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(2).locator('[data-testid="select-checkbox"]').check();
    
    // Enter comparison mode
    await page.click('[data-testid="compare-button"]');
    
    // Verify all three gifts are displayed
    const comparisonCards = page.locator('[data-testid="comparison-card"]');
    await expect(comparisonCards).toHaveCount(3);
    
    // Verify comparison chart handles three datasets
    await expect(page.locator('[data-testid="comparison-category-chart"]')).toBeVisible();
  });

  test('should deselect gift in comparison mode', async ({ page }) => {
    // Select two gifts and enter comparison mode
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    await page.click('[data-testid="compare-button"]');
    
    // Verify two gifts in comparison
    await expect(page.locator('[data-testid="comparison-card"]')).toHaveCount(2);
    
    // Deselect one gift
    await page.locator('[data-testid="comparison-card"]').first().locator('[data-testid="remove-from-comparison"]').click();
    
    // Verify only one gift remains
    await expect(page.locator('[data-testid="comparison-card"]')).toHaveCount(1);
  });
});
