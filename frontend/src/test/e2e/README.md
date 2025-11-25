# End-to-End Tests

This directory contains Playwright E2E tests for the reasoning visualization feature.

## Running Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run tests in UI mode
npm run test:e2e:ui

# Run tests in debug mode
npm run test:e2e:debug

# Run specific test file
npx playwright test reasoning-flow.spec.ts

# Run tests on specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit

# Run mobile tests
npx playwright test --project="Mobile Chrome"
npx playwright test --project="Mobile Safari"
```

## Test Structure

### 1. reasoning-flow.spec.ts
Tests the complete user journey through reasoning visualization:
- User profile submission
- Viewing recommendations with reasoning
- Opening detailed reasoning panel
- Interacting with reasoning sections (tool selection, category matching, attention weights, thinking steps)
- Confidence indicator and explanation modal
- Reasoning level persistence
- Filter management
- Loading and error states

**Key Test Cases:**
- Full reasoning visualization flow (end-to-end)
- Reasoning level persistence (localStorage)
- Section filtering (tool selection, category matching, etc.)
- Loading states and error handling

### 2. comparison-mode.spec.ts
Tests the comparison functionality:
- Selecting multiple gifts
- Activating comparison mode
- Side-by-side reasoning display
- Category score comparison charts
- Attention weights overlay charts
- Exiting comparison mode
- Handling 3+ gift comparisons

**Key Test Cases:**
- Enable comparison with 2+ gifts
- Side-by-side gift display
- Comparison charts with different colors
- Exit comparison and return to normal view
- Deselect gifts in comparison mode

### 3. mobile-responsive.spec.ts
Tests responsive design across different viewports:
- Mobile phone (375px - iPhone 12)
- Tablet (768px - iPad)
- Desktop (1920px)
- Portrait/landscape orientation changes
- Touch interactions and gestures
- Swipe to close panels
- Touch-friendly tooltips

**Key Test Cases:**
- Mobile layout adaptation (vertical stacking)
- Vertical chart layouts on mobile
- Full-screen modals on mobile
- Swipe gestures
- Touch-friendly UI elements
- Tablet side panel layout
- Desktop multi-column grid

### 4. accessibility.spec.ts
Tests accessibility compliance:
- ARIA labels and roles
- Keyboard navigation
- Focus management
- Screen reader compatibility
- Color contrast (WCAG 2.0 AA)
- Color-blind friendly design
- Heading hierarchy
- Reduced motion support

**Key Test Cases:**
- Automated accessibility checks (axe-core)
- Complete keyboard navigation
- Focus trap in modals
- Screen reader announcements
- Proper heading hierarchy
- Color contrast compliance
- Alternative text for images
- Skip links for keyboard users

### 5. export-functionality.spec.ts
Tests export features:
- JSON export with proper structure
- PDF export with visualizations
- Share link copy to clipboard
- Success notifications
- Error handling
- Metadata inclusion
- Unique filename generation

**Key Test Cases:**
- Export reasoning as JSON
- Export reasoning as PDF
- Copy share link to clipboard
- Success messages for each export type
- Error handling for failed exports
- Export with filtered sections
- Export comparison data
- Clipboard permission handling

## Test Data

Tests use the following data-testid attributes:
- `gift-card` - Gift recommendation card
- `reasoning-text` - Reasoning explanation text
- `confidence-indicator` - Confidence score indicator
- `show-details-button` - Button to open reasoning panel
- `reasoning-panel` - Detailed reasoning panel
- `tool-selection-section` - Tool selection visualization
- `category-matching-section` - Category matching chart
- `attention-weights-section` - Attention weights chart
- `thinking-steps-section` - Thinking steps timeline
- `confidence-modal` - Confidence explanation modal
- `comparison-view` - Comparison mode view
- `export-button` - Export dropdown button
- `export-json` - JSON export option
- `export-pdf` - PDF export option
- `export-share` - Share link option

## Prerequisites

Before running E2E tests, ensure:
1. Development server is running (or will be started automatically)
2. Backend API is accessible
3. Test data is available
4. Required browsers are installed: `npx playwright install`

## CI/CD Integration

E2E tests are configured to run in CI with:
- Retry on failure (2 retries)
- Single worker (no parallel execution)
- HTML report generation
- Screenshots on failure
- Trace on first retry

## Debugging

### Visual Debugging
```bash
# Open Playwright Inspector
npm run test:e2e:debug

# Run with headed browser
npx playwright test --headed

# Run with slow motion
npx playwright test --headed --slow-mo=1000
```

### Screenshots and Videos
Failed tests automatically capture:
- Screenshots (on failure)
- Traces (on first retry)
- Videos (optional, configure in playwright.config.ts)

### View Test Report
```bash
# Generate and open HTML report
npx playwright show-report
```

## Best Practices

1. **Use data-testid attributes** for stable selectors
2. **Wait for network idle** before assertions
3. **Test user flows**, not implementation details
4. **Keep tests independent** - each test should work in isolation
5. **Use page object pattern** for complex pages (if needed)
6. **Mock external APIs** when appropriate
7. **Test accessibility** in every flow
8. **Verify responsive behavior** across viewports

## Troubleshooting

### Tests timing out
- Increase timeout in playwright.config.ts
- Check if dev server is running
- Verify network requests complete

### Flaky tests
- Add explicit waits for elements
- Use `waitForLoadState('networkidle')`
- Check for race conditions

### Accessibility violations
- Review axe-core report
- Fix ARIA labels and roles
- Ensure proper color contrast
- Add keyboard navigation support

## Coverage

E2E tests cover:
- ✅ Full reasoning flow (Requirements 1-7)
- ✅ Comparison mode (Requirements 12)
- ✅ Mobile responsive (Requirements 10)
- ✅ Accessibility (Requirements 15)
- ✅ Export functionality (Requirements 14)
- ✅ Loading/error states (Requirements 9)
- ✅ Filter management (Requirements 11)

## Future Enhancements

Potential additions:
- Performance testing (Lighthouse CI)
- Visual regression testing
- API mocking for offline testing
- Cross-browser compatibility matrix
- Internationalization testing
