import { test, expect } from '@playwright/test';

/**
 * E2E Test: Full Reasoning Flow
 * 
 * This test validates the complete user journey through the reasoning visualization:
 * 1. User fills in profile
 * 2. Gets recommendations with reasoning
 * 3. Views basic reasoning on cards
 * 4. Opens detailed reasoning panel
 * 5. Explores different reasoning sections
 */

test.describe('Full Reasoning Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to home page
    await page.goto('/');
  });

  test('should complete full reasoning visualization flow', async ({ page }) => {
    // Step 1: Fill in user profile
    await page.fill('input[name="age"]', '25');
    await page.fill('input[name="budget"]', '500');
    await page.selectOption('select[name="occasion"]', 'birthday');
    await page.fill('input[name="hobbies"]', 'reading, gaming');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Step 2: Wait for recommendations to load
    await expect(page.locator('[data-testid="gift-card"]').first()).toBeVisible({ timeout: 10000 });
    
    // Step 3: Verify basic reasoning is displayed on card
    const firstCard = page.locator('[data-testid="gift-card"]').first();
    await expect(firstCard.locator('[data-testid="reasoning-text"]')).toBeVisible();
    await expect(firstCard.locator('[data-testid="confidence-indicator"]')).toBeVisible();
    
    // Step 4: Verify confidence indicator shows correct styling
    const confidenceIndicator = firstCard.locator('[data-testid="confidence-indicator"]');
    const confidenceText = await confidenceIndicator.textContent();
    expect(confidenceText).toMatch(/(Yüksek Güven|Orta Güven|Düşük Güven)/);
    
    // Step 5: Click "Show Details" button to open reasoning panel
    await firstCard.locator('[data-testid="show-details-button"]').click();
    
    // Step 6: Verify reasoning panel opens
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
    
    // Step 7: Verify all reasoning sections are present
    await expect(page.locator('[data-testid="tool-selection-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-matching-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="attention-weights-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="thinking-steps-section"]')).toBeVisible();
    
    // Step 8: Interact with tool selection
    const toolCard = page.locator('[data-testid="tool-card"]').first();
    await toolCard.hover();
    await expect(page.locator('[role="tooltip"]')).toBeVisible();
    
    // Step 9: Interact with category matching chart
    const categoryBar = page.locator('[data-testid="category-bar"]').first();
    await categoryBar.click();
    await expect(page.locator('[data-testid="category-reasons"]')).toBeVisible();
    
    // Step 10: Switch attention weights chart type
    await page.click('[data-testid="chart-type-toggle"]');
    await expect(page.locator('[data-testid="radar-chart"]')).toBeVisible();
    
    // Step 11: Expand thinking step
    const thinkingStep = page.locator('[data-testid="thinking-step"]').first();
    await thinkingStep.click();
    await expect(page.locator('[data-testid="step-details"]').first()).toBeVisible();
    
    // Step 12: Click confidence indicator to open explanation modal
    await page.click('[data-testid="confidence-indicator"]');
    await expect(page.locator('[data-testid="confidence-modal"]')).toBeVisible();
    await expect(page.locator('[data-testid="positive-factors"]')).toBeVisible();
    await expect(page.locator('[data-testid="negative-factors"]')).toBeVisible();
    
    // Step 13: Close modal
    await page.click('[data-testid="close-modal-button"]');
    await expect(page.locator('[data-testid="confidence-modal"]')).not.toBeVisible();
    
    // Step 14: Close reasoning panel
    await page.click('[data-testid="close-panel-button"]');
    await expect(page.locator('[data-testid="reasoning-panel"]')).not.toBeVisible();
  });

  test('should handle reasoning level persistence', async ({ page }) => {
    // Navigate to recommendations page
    await page.goto('/recommendations');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Change reasoning level to 'detailed'
    await page.click('[data-testid="reasoning-level-selector"]');
    await page.click('[data-testid="reasoning-level-detailed"]');
    
    // Verify localStorage is updated
    const reasoningLevel = await page.evaluate(() => localStorage.getItem('reasoningLevel'));
    expect(reasoningLevel).toBe('detailed');
    
    // Reload page
    await page.reload();
    
    // Verify reasoning level persists
    const persistedLevel = await page.evaluate(() => localStorage.getItem('reasoningLevel'));
    expect(persistedLevel).toBe('detailed');
  });

  test('should filter reasoning sections', async ({ page }) => {
    // Navigate to recommendations with reasoning
    await page.goto('/recommendations');
    await page.waitForLoadState('networkidle');
    
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
    
    // Test "Only Tool Selection" filter
    await page.click('[data-testid="filter-selector"]');
    await page.click('[data-testid="filter-tool-selection"]');
    
    await expect(page.locator('[data-testid="tool-selection-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-matching-section"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="attention-weights-section"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="thinking-steps-section"]')).not.toBeVisible();
    
    // Test "Only Category Matching" filter
    await page.click('[data-testid="filter-selector"]');
    await page.click('[data-testid="filter-category-matching"]');
    
    await expect(page.locator('[data-testid="tool-selection-section"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="category-matching-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="attention-weights-section"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="thinking-steps-section"]')).not.toBeVisible();
    
    // Test "Show All" filter
    await page.click('[data-testid="filter-selector"]');
    await page.click('[data-testid="filter-show-all"]');
    
    await expect(page.locator('[data-testid="tool-selection-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-matching-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="attention-weights-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="thinking-steps-section"]')).toBeVisible();
  });

  test('should handle loading and error states', async ({ page }) => {
    // Navigate to recommendations
    await page.goto('/recommendations');
    
    // Verify loading state appears
    await expect(page.locator('[data-testid="skeleton-loader"]')).toBeVisible();
    
    // Wait for content to load
    await page.waitForLoadState('networkidle');
    
    // Verify loading state disappears
    await expect(page.locator('[data-testid="skeleton-loader"]')).not.toBeVisible();
    
    // Simulate error by intercepting API call
    await page.route('**/api/recommendations', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });
    
    // Trigger new request
    await page.click('[data-testid="retry-button"]');
    
    // Verify error message appears
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
  });
});
