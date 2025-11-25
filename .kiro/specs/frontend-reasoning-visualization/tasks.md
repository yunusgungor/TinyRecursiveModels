# Implementation Plan

- [x] 1. Project Setup and Configuration
  - Install required dependencies (Recharts, Radix UI, fast-check, Playwright)
  - Configure Vitest for unit and property-based testing
  - Set up TypeScript interfaces for reasoning data models
  - Configure environment variables for API integration
  - Set up feature flags for reasoning functionality
  - _Requirements: 8.6_

- [x] 2. Core Data Models and Types
  - Create TypeScript interfaces for ToolSelectionReasoning
  - Create TypeScript interfaces for CategoryMatchingReasoning
  - Create TypeScript interfaces for AttentionWeights
  - Create TypeScript interfaces for ThinkingStep
  - Create TypeScript interfaces for ConfidenceExplanation
  - Create TypeScript interfaces for ReasoningTrace
  - Create TypeScript interfaces for EnhancedGiftRecommendation
  - Create TypeScript interfaces for EnhancedRecommendationResponse
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 3. API Service Layer
  - Implement RecommendationAPIClient class
  - Implement fetchRecommendations method with reasoning parameters
  - Implement error handling and retry logic
  - Implement request cancellation support
  - _Requirements: 9.2, 9.5_

- [x] 3.1 Write unit tests for API service
  - Test fetchRecommendations with different parameters
  - Test error handling scenarios
  - Test request cancellation

- [x] 4. Custom Hooks Implementation
  - Implement useRecommendations hook
  - Implement useReasoningPanel hook
  - Implement useReasoningLevel hook with localStorage persistence
  - Implement useMediaQuery hook for responsive design
  - Implement useKeyboardNavigation hook for accessibility
  - _Requirements: 7.5, 7.6, 10.1, 15.3_

- [x] 4.1 Write property test for reasoning level persistence
  - **Property 35: Reasoning level persistence (round-trip)**
  - **Validates: Requirements 7.5, 7.6**

- [x] 4.2 Write unit tests for custom hooks
  - Test useRecommendations loading and error states
  - Test useReasoningPanel state management
  - Test useReasoningLevel localStorage integration

- [x] 5. Confidence Indicator Component
  - Create ConfidenceIndicator component
  - Implement color coding (green >0.8, yellow 0.5-0.8, red <0.5)
  - Implement confidence label display
  - Add click handler for explanation modal
  - Add ARIA labels for accessibility
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 15.1_

- [x] 5.1 Write property test for confidence styling
  - **Property 27: High confidence styling**
  - **Property 28: Medium confidence styling**
  - **Property 29: Low confidence styling**
  - **Validates: Requirements 6.2, 6.3, 6.4**

- [x] 5.2 Write unit tests for ConfidenceIndicator
  - Test rendering with different confidence values
  - Test click handler invocation
  - Test ARIA attributes

- [x] 6. Confidence Explanation Modal Component
  - Create ConfidenceExplanationModal component
  - Implement modal open/close functionality
  - Display confidence score and level
  - Render positive factors with green styling
  - Render negative factors with red styling
  - Add keyboard navigation support
  - _Requirements: 6.5, 6.6, 15.3_

- [x] 6.1 Write property test for confidence explanation modal
  - **Property 30: Confidence explanation modal**
  - **Property 31: Factor categorization**
  - **Validates: Requirements 6.5, 6.6**

- [x] 7. Gift Recommendation Card Component
  - Create GiftRecommendationCard component
  - Display gift information (name, price, image)
  - Render reasoning strings with highlighting
  - Implement expandable reasoning text with "Show more" button
  - Display tool insights as icons (rating, trend, availability)
  - Integrate ConfidenceIndicator
  - Add "Show Details" button
  - Add selection checkbox for comparison mode
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 7.1 Write property test for reasoning display
  - **Property 1: Reasoning display completeness**
  - **Property 3: Tool insights icon rendering**
  - **Property 4: Expandable reasoning text**
  - **Validates: Requirements 1.1, 1.3, 1.4**

- [x] 7.2 Write unit tests for GiftRecommendationCard
  - Test rendering with various gift data
  - Test expand/collapse functionality
  - Test button click handlers

- [x] 8. Tool Selection Card Component
  - Create ToolSelectionCard component
  - Render tool list with selection status
  - Style selected tools with green color and checkmark
  - Style unselected tools with gray color
  - Display confidence scores and priority
  - Implement hover tooltips with reasons and factors
  - Add low confidence tooltips
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 8.1 Write property test for tool selection styling
  - **Property 7: Selected tool styling**
  - **Property 8: Unselected tool styling**
  - **Property 9: Low confidence tooltip**
  - **Validates: Requirements 2.3, 2.4, 2.5**

- [x] 8.2 Write unit tests for ToolSelectionCard
  - Test rendering with different tool states
  - Test tooltip display on hover
  - Test accessibility attributes

- [x] 9. Category Matching Chart Component
  - Create CategoryMatchingChart component using Recharts
  - Render horizontal bar chart with category scores
  - Implement color coding (green >0.7, red <0.3, yellow medium)
  - Display at least top 3 categories
  - Format scores as percentages
  - Implement click handler to show reasons
  - Add tooltips for additional information
  - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 9.1 Write property test for category matching
  - **Property 11: Minimum category count**
  - **Property 12: High score category styling**
  - **Property 13: Low score category styling**
  - **Property 15: Score percentage formatting**
  - **Validates: Requirements 3.2, 3.3, 3.4, 3.6**

- [x] 9.2 Write unit tests for CategoryMatchingChart
  - Test rendering with various score ranges
  - Test click expansion functionality
  - Test chart accessibility

- [x] 10. Attention Weights Chart Component
  - Create AttentionWeightsChart component using Recharts
  - Implement bar chart view for user and gift features
  - Implement radar chart view as alternative
  - Add chart type toggle button
  - Display weights as percentages
  - Implement hover tooltips with full values
  - Ensure charts are responsive
  - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 10.1 Write property test for attention weights
  - **Property 18: Weight percentage display**
  - **Property 20: Chart type switching**
  - **Validates: Requirements 4.4, 4.6**

- [x] 10.2 Write unit tests for AttentionWeightsChart
  - Test bar chart rendering
  - Test radar chart rendering
  - Test chart type toggle
  - Test tooltip display

- [x] 11. Thinking Steps Timeline Component
  - Create ThinkingStepsTimeline component
  - Render vertical timeline with chronological ordering
  - Display step number, action, result, and insight
  - Mark completed steps with green checkmarks
  - Implement click to expand step details
  - Add scrollable container for long timelines
  - Implement keyboard navigation
  - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 11.1 Write property test for thinking steps
  - **Property 21: Chronological step ordering**
  - **Property 22: Step information completeness**
  - **Property 23: Completed step marking**
  - **Validates: Requirements 5.2, 5.3, 5.4**

- [x] 11.2 Write unit tests for ThinkingStepsTimeline
  - Test rendering with different step counts
  - Test expand/collapse functionality
  - Test keyboard navigation

- [x] 12. Reasoning Panel Component
  - Create ReasoningPanel component
  - Implement panel open/close functionality
  - Add filter selection UI (tool selection, category matching, attention weights, thinking steps)
  - Conditionally render sections based on active filters
  - Integrate all reasoning sub-components
  - Add export button with dropdown (JSON, PDF, Share)
  - Implement responsive layout (full-screen on mobile)
  - Add swipe gesture support for mobile
  - _Requirements: 7.2, 7.3, 7.4, 10.3, 10.4, 11.2, 11.3, 11.4, 11.5_

- [x] 12.1 Write property test for filter management
  - **Property 45: Tool selection filter**
  - **Property 46: Category matching filter**
  - **Property 47: Attention weights filter**
  - **Property 48: Show all filter**
  - **Validates: Requirements 11.2, 11.3, 11.4, 11.5**

- [x] 12.2 Write unit tests for ReasoningPanel
  - Test panel open/close
  - Test filter functionality
  - Test mobile responsive behavior

- [x] 13. Export Functionality
  - Implement exportReasoningAsJSON function
  - Implement exportReasoningAsPDF function using jsPDF
  - Implement copyReasoningLink function
  - Add success toast notifications
  - Handle export errors gracefully
  - _Requirements: 14.2, 14.3, 14.4, 14.5_

- [x] 13.1 Write property test for export functionality
  - **Property 54: JSON export**
  - **Property 56: Share link copy**
  - **Property 57: Export success message**
  - **Validates: Requirements 14.2, 14.4, 14.5**

- [x] 13.2 Write unit tests for export functions
  - Test JSON export with various data
  - Test PDF generation
  - Test clipboard copy

- [x] 14. Comparison Mode Implementation
  - Add gift selection state management
  - Implement "Compare" button display logic
  - Create comparison view layout (side-by-side)
  - Implement comparison charts for categories
  - Implement overlay charts for attention weights
  - Add comparison mode exit functionality
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 14.1 Write property test for comparison mode
  - **Property 49: Compare button display**
  - **Property 50: Side-by-side comparison**
  - **Property 53: Comparison mode exit**
  - **Validates: Requirements 12.1, 12.2, 12.5**

- [x] 14.2 Write unit tests for comparison mode
  - Test gift selection logic
  - Test comparison view rendering
  - Test comparison charts

- [x] 15. Loading and Error States
  - Create skeleton loaders for reasoning components
  - Implement loading spinners for API requests
  - Create error message components
  - Add retry button functionality
  - Implement request cancellation on user action
  - _Requirements: 9.1, 9.2, 9.4, 9.5_

- [x] 15.1 Write property test for loading and error states
  - **Property 36: Loading state display**
  - **Property 37: Error message display**
  - **Property 38: Retry button display**
  - **Validates: Requirements 9.1, 9.2, 9.4**

- [x] 15.2 Write unit tests for loading and error states
  - Test skeleton loader rendering
  - Test error message display
  - Test retry functionality

- [x] 16. Responsive Design Implementation
  - Implement mobile layout adaptations
  - Add vertical chart layout for mobile (<768px)
  - Implement full-screen modal for mobile
  - Add touch gesture support (swipe to close)
  - Implement touch-friendly tooltips
  - Test on various screen sizes
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 16.1 Write property test for responsive design
  - **Property 41: Vertical chart layout**
  - **Property 42: Full-screen mobile modal**
  - **Property 43: Swipe gesture support**
  - **Validates: Requirements 10.2, 10.3, 10.4**

- [x] 16.2 Write integration tests for responsive behavior
  - Test mobile viewport rendering
  - Test tablet viewport rendering
  - Test desktop viewport rendering

- [x] 17. Accessibility Implementation
  - Add ARIA labels and roles to all components
  - Implement alt text for charts
  - Ensure keyboard navigation works throughout
  - Test with screen readers
  - Implement color-blind friendly design (patterns, icons)
  - Add focus management
  - Test color contrast ratios
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [x] 17.1 Write property test for accessibility
  - **Property 58: ARIA labels presence**
  - **Property 60: Keyboard navigation**
  - **Property 62: Color-blind friendly design**
  - **Validates: Requirements 15.1, 15.3, 15.5**

- [x] 17.2 Write accessibility tests
  - Test keyboard navigation flow
  - Test screen reader compatibility
  - Test focus management

- [x] 18. Reasoning Context and State Management
  - Create ReasoningContext with Provider
  - Implement reasoning level state management
  - Implement gift selection state management
  - Implement comparison mode state management
  - Add localStorage persistence for preferences
  - _Requirements: 7.5, 7.6, 12.1_

- [x] 18.1 Write unit tests for context
  - Test context provider
  - Test state updates
  - Test localStorage persistence

- [x] 19. Recommendations Page Integration
  - Create or update RecommendationsPage
  - Integrate useRecommendations hook
  - Render GiftRecommendationCard list
  - Implement ReasoningPanel integration
  - Add comparison mode UI
  - Handle loading and error states
  - _Requirements: 1.1, 7.1, 7.2_

- [x] 19.1 Write integration tests for recommendations page
  - Test full user flow
  - Test reasoning panel interaction
  - Test comparison mode

- [x] 20. Performance Optimizations
  - Implement lazy loading for heavy components
  - Add React.memo to expensive components
  - Implement virtual scrolling for long lists
  - Optimize chart rendering
  - Add code splitting for reasoning features
  - _Requirements: Performance considerations_

- [x] 20.1 Write performance tests
  - Measure component render times
  - Test virtual scrolling performance
  - Test lazy loading behavior

- [x] 21. Storybook Documentation
  - Create Storybook stories for all components
  - Add usage examples and variations
  - Document component props and behaviors
  - Add accessibility notes
  - _Requirements: 8.1_

- [x] 22. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 23. End-to-End Testing with Playwright
  - Write E2E test for full reasoning flow
  - Write E2E test for comparison mode
  - Write E2E test for mobile responsive behavior
  - Write E2E test for accessibility flow
  - Write E2E test for export functionality

- [x] 24. Final Integration and Polish
  - Test complete user journey
  - Fix any remaining bugs
  - Optimize bundle size
  - Update documentation
  - Prepare for deployment
