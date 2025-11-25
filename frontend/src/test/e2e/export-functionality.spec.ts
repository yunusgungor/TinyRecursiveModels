import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * E2E Test: Export Functionality
 * 
 * This test validates export features:
 * 1. JSON export
 * 2. PDF export
 * 3. Share link copy
 * 4. Success notifications
 * 5. Error handling
 */

test.describe('Export Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/recommendations');
    await page.waitForLoadState('networkidle');
    
    // Open reasoning panel
    await page.locator('[data-testid="show-details-button"]').first().click();
    await expect(page.locator('[data-testid="reasoning-panel"]')).toBeVisible();
  });

  test('should export reasoning data as JSON', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Click export button
    await page.click('[data-testid="export-button"]');
    
    // Select JSON export option
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    const download = await downloadPromise;
    
    // Verify download filename
    const filename = download.suggestedFilename();
    expect(filename).toMatch(/reasoning.*\.json$/);
    
    // Save and verify file content
    const downloadPath = path.join(__dirname, 'downloads', filename);
    await download.saveAs(downloadPath);
    
    // Read and parse JSON
    const fileContent = fs.readFileSync(downloadPath, 'utf-8');
    const jsonData = JSON.parse(fileContent);
    
    // Verify JSON structure
    expect(jsonData).toHaveProperty('tool_selection');
    expect(jsonData).toHaveProperty('category_matching');
    expect(jsonData).toHaveProperty('attention_weights');
    expect(jsonData).toHaveProperty('thinking_steps');
    
    // Verify data is not empty
    expect(Array.isArray(jsonData.tool_selection)).toBeTruthy();
    expect(Array.isArray(jsonData.category_matching)).toBeTruthy();
    expect(Array.isArray(jsonData.thinking_steps)).toBeTruthy();
    
    // Clean up
    fs.unlinkSync(downloadPath);
  });

  test('should export reasoning visualizations as PDF', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Click export button
    await page.click('[data-testid="export-button"]');
    
    // Select PDF export option
    await page.click('[data-testid="export-pdf"]');
    
    // Wait for download
    const download = await downloadPromise;
    
    // Verify download filename
    const filename = download.suggestedFilename();
    expect(filename).toMatch(/reasoning.*\.pdf$/);
    
    // Save file
    const downloadPath = path.join(__dirname, 'downloads', filename);
    await download.saveAs(downloadPath);
    
    // Verify file exists and has content
    const stats = fs.statSync(downloadPath);
    expect(stats.size).toBeGreaterThan(1000); // PDF should be at least 1KB
    
    // Verify it's a valid PDF (starts with %PDF)
    const buffer = fs.readFileSync(downloadPath);
    const header = buffer.toString('utf-8', 0, 4);
    expect(header).toBe('%PDF');
    
    // Clean up
    fs.unlinkSync(downloadPath);
  });

  test('should copy reasoning link to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    
    // Click export button
    await page.click('[data-testid="export-button"]');
    
    // Select share option
    await page.click('[data-testid="export-share"]');
    
    // Wait for clipboard operation
    await page.waitForTimeout(500);
    
    // Read clipboard content
    const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
    
    // Verify clipboard contains a valid URL
    expect(clipboardText).toBeTruthy();
    expect(clipboardText).toMatch(/^https?:\/\//);
    expect(clipboardText).toContain('reasoning');
    
    // Verify success message appears
    await expect(page.locator('[data-testid="success-toast"]')).toBeVisible();
    const toastText = await page.locator('[data-testid="success-toast"]').textContent();
    expect(toastText).toContain('kopyalandı');
  });

  test('should display success message after JSON export', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Click export button and select JSON
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    await downloadPromise;
    
    // Verify success toast appears
    await expect(page.locator('[data-testid="success-toast"]')).toBeVisible();
    
    // Verify success message content
    const toastText = await page.locator('[data-testid="success-toast"]').textContent();
    expect(toastText).toContain('başarıyla');
    expect(toastText).toContain('JSON');
  });

  test('should display success message after PDF export', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Click export button and select PDF
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-pdf"]');
    
    // Wait for download
    await downloadPromise;
    
    // Verify success toast appears
    await expect(page.locator('[data-testid="success-toast"]')).toBeVisible();
    
    // Verify success message content
    const toastText = await page.locator('[data-testid="success-toast"]').textContent();
    expect(toastText).toContain('başarıyla');
    expect(toastText).toContain('PDF');
  });

  test('should handle export errors gracefully', async ({ page }) => {
    // Intercept export request and simulate error
    await page.route('**/api/export/**', route => {
      route.abort('failed');
    });
    
    // Try to export
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-json"]');
    
    // Verify error message appears
    await expect(page.locator('[data-testid="error-toast"]')).toBeVisible();
    
    // Verify error message content
    const errorText = await page.locator('[data-testid="error-toast"]').textContent();
    expect(errorText).toContain('hata');
  });

  test('should export with filtered reasoning sections', async ({ page }) => {
    // Apply filter to show only tool selection
    await page.click('[data-testid="filter-selector"]');
    await page.click('[data-testid="filter-tool-selection"]');
    
    // Verify only tool selection is visible
    await expect(page.locator('[data-testid="tool-selection-section"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-matching-section"]')).not.toBeVisible();
    
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Export as JSON
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    const download = await downloadPromise;
    const downloadPath = path.join(__dirname, 'downloads', download.suggestedFilename());
    await download.saveAs(downloadPath);
    
    // Read and verify JSON includes all data (not filtered)
    const fileContent = fs.readFileSync(downloadPath, 'utf-8');
    const jsonData = JSON.parse(fileContent);
    
    // Export should include all data regardless of UI filters
    expect(jsonData).toHaveProperty('tool_selection');
    expect(jsonData).toHaveProperty('category_matching');
    expect(jsonData).toHaveProperty('attention_weights');
    expect(jsonData).toHaveProperty('thinking_steps');
    
    // Clean up
    fs.unlinkSync(downloadPath);
  });

  test('should export comparison data for multiple gifts', async ({ page }) => {
    // Close reasoning panel
    await page.click('[data-testid="close-panel-button"]');
    
    // Select multiple gifts
    await page.locator('[data-testid="gift-card"]').first().locator('[data-testid="select-checkbox"]').check();
    await page.locator('[data-testid="gift-card"]').nth(1).locator('[data-testid="select-checkbox"]').check();
    
    // Enter comparison mode
    await page.click('[data-testid="compare-button"]');
    await expect(page.locator('[data-testid="comparison-view"]')).toBeVisible();
    
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Export comparison as JSON
    await page.click('[data-testid="export-comparison-button"]');
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    const download = await downloadPromise;
    const downloadPath = path.join(__dirname, 'downloads', download.suggestedFilename());
    await download.saveAs(downloadPath);
    
    // Read and verify JSON includes comparison data
    const fileContent = fs.readFileSync(downloadPath, 'utf-8');
    const jsonData = JSON.parse(fileContent);
    
    // Should have data for multiple gifts
    expect(jsonData).toHaveProperty('gifts');
    expect(Array.isArray(jsonData.gifts)).toBeTruthy();
    expect(jsonData.gifts.length).toBeGreaterThanOrEqual(2);
    
    // Clean up
    fs.unlinkSync(downloadPath);
  });

  test('should handle clipboard permission denial', async ({ page, context }) => {
    // Deny clipboard permissions
    await context.grantPermissions([]);
    
    // Try to copy link
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-share"]');
    
    // Should show error or fallback message
    const toast = page.locator('[data-testid="error-toast"], [data-testid="info-toast"]');
    await expect(toast).toBeVisible();
  });

  test('should include metadata in exported JSON', async ({ page }) => {
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Export as JSON
    await page.click('[data-testid="export-button"]');
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    const download = await downloadPromise;
    const downloadPath = path.join(__dirname, 'downloads', download.suggestedFilename());
    await download.saveAs(downloadPath);
    
    // Read and verify JSON includes metadata
    const fileContent = fs.readFileSync(downloadPath, 'utf-8');
    const jsonData = JSON.parse(fileContent);
    
    // Should include metadata
    expect(jsonData).toHaveProperty('metadata');
    expect(jsonData.metadata).toHaveProperty('export_date');
    expect(jsonData.metadata).toHaveProperty('gift_id');
    
    // Clean up
    fs.unlinkSync(downloadPath);
  });

  test('should generate unique filenames for multiple exports', async ({ page }) => {
    const filenames: string[] = [];
    
    // Export multiple times
    for (let i = 0; i < 3; i++) {
      const downloadPromise = page.waitForEvent('download');
      
      await page.click('[data-testid="export-button"]');
      await page.click('[data-testid="export-json"]');
      
      const download = await downloadPromise;
      filenames.push(download.suggestedFilename());
      
      // Wait a bit between exports
      await page.waitForTimeout(100);
    }
    
    // Verify all filenames are unique
    const uniqueFilenames = new Set(filenames);
    expect(uniqueFilenames.size).toBe(filenames.length);
  });

  test('should close export menu after selection', async ({ page }) => {
    // Open export menu
    await page.click('[data-testid="export-button"]');
    
    // Verify menu is visible
    await expect(page.locator('[data-testid="export-menu"]')).toBeVisible();
    
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    
    // Select export option
    await page.click('[data-testid="export-json"]');
    
    // Wait for download
    await downloadPromise;
    
    // Verify menu closes
    await expect(page.locator('[data-testid="export-menu"]')).not.toBeVisible();
  });
});

// Create downloads directory if it doesn't exist
test.beforeAll(() => {
  const downloadsDir = path.join(__dirname, 'downloads');
  if (!fs.existsSync(downloadsDir)) {
    fs.mkdirSync(downloadsDir, { recursive: true });
  }
});

// Clean up downloads directory after all tests
test.afterAll(() => {
  const downloadsDir = path.join(__dirname, 'downloads');
  if (fs.existsSync(downloadsDir)) {
    const files = fs.readdirSync(downloadsDir);
    files.forEach(file => {
      fs.unlinkSync(path.join(downloadsDir, file));
    });
    fs.rmdirSync(downloadsDir);
  }
});
