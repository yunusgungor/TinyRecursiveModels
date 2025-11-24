# Implementation Plan

- [x] 1. Model Layer: Attention Weights Extraction
  - Implement `extract_attention_weights()` method in IntegratedEnhancedTRM
  - Extract user features attention (hobbies, budget, age, occasion)
  - Extract gift features attention (category, price, rating)
  - Normalize weights to sum to 1.0
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 1.1 Write property test for attention weights normalization
  - **Property 19: Attention weights normalization**
  - **Validates: Requirements 4.4**

- [x] 2. Model Layer: Thinking Steps Generation
  - Implement `get_thinking_steps()` method in IntegratedEnhancedTRM
  - Record user encoding step with insight
  - Record category matching step with top categories
  - Record tool selection step with selected tools
  - Record tool execution step with results summary
  - Record gift ranking step with criteria
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2.1 Write property test for thinking steps structure
  - **Property 22: Thinking step structure**
  - **Validates: Requirements 5.2**

- [x] 2.2 Write property test for chronological ordering
  - **Property 26: Chronological step ordering**
  - **Validates: Requirements 5.6**

- [x] 3. Model Layer: Tool Selection Explanation
  - Implement `explain_tool_selection()` method in IntegratedEnhancedTRM
  - Generate selection reason for each tool based on user profile
  - Include confidence score and priority for selected tools
  - Identify user profile features that influenced selection
  - Handle low confidence cases with explanations
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3.1 Write property test for tool selection reasoning completeness
  - **Property 1: Tool selection reasoning completeness**
  - **Validates: Requirements 1.1**

- [x] 3.2 Write property test for low confidence explanation
  - **Property 4: Low confidence tool selection explanation**
  - **Validates: Requirements 1.4**

- [x] 4. Model Layer: Category Matching Explanation
  - Implement `explain_category_matching()` method in IntegratedEnhancedTRM
  - Compute category scores and contributing factors
  - Generate explanations for high score categories (>0.7)
  - Generate explanations for low score categories (<0.3)
  - Return at least top 3 categories with scores
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4.1 Write property test for category score and explanation consistency
  - **Property 11: Category score and explanation consistency**
  - **Validates: Requirements 9.3**

- [x] 4.2 Write property test for top categories minimum count
  - **Property 9: Top categories minimum count**
  - **Validates: Requirements 2.4**

- [x] 5. Model Layer: Forward Pass with Reasoning Trace
  - Implement `forward_with_reasoning_trace()` wrapper method
  - Call existing `forward_with_enhancements()` method
  - Extract reasoning components when capture_reasoning=True
  - Return (carry, rl_output, selected_tools, reasoning_trace)
  - Add helper methods for summarizing user profile and tool results
  - _Requirements: 1.1, 2.1, 4.1, 5.1_

- [x] 6. Backend: ReasoningService Implementation
  - Create new `ReasoningService` class in `backend/app/services/reasoning_service.py`
  - Implement `generate_gift_reasoning()` for dynamic reasoning
  - Implement `explain_confidence_score()` for confidence explanations
  - Implement `generate_tool_selection_reasoning()` for tool reasoning
  - Implement `generate_category_reasoning()` for category reasoning
  - Load hobby-category and occasion-category mappings
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3_

- [x] 6.1 Write property test for dynamic gift reasoning
  - **Property 12: Dynamic gift reasoning generation**
  - **Validates: Requirements 3.1, 3.6**

- [x] 6.2 Write property test for hobby matching explanation
  - **Property 13: Hobby matching explanation**
  - **Validates: Requirements 3.2**

- [x] 6.3 Write property test for confidence threshold differentiation
  - **Property 30: Confidence threshold differentiation**
  - **Validates: Requirements 6.5**

- [x] 7. Backend: Enhanced Response Schemas
  - Create `ToolSelectionReasoning` Pydantic model
  - Create `CategoryMatchingReasoning` Pydantic model
  - Create `AttentionWeights` Pydantic model
  - Create `ThinkingStep` Pydantic model
  - Create `ConfidenceExplanation` Pydantic model
  - Create `ReasoningTrace` Pydantic model
  - Create `EnhancedGiftRecommendation` model extending GiftRecommendation
  - Create `EnhancedRecommendationResponse` model
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 7.1 Write property test for JSON schema compliance
  - **Property 31: Reasoning JSON schema compliance**
  - **Validates: Requirements 8.1**

- [x] 7.2 Write property test for tool selection schema
  - **Property 5: Tool selection JSON schema compliance**
  - **Validates: Requirements 1.5, 8.2**

- [x] 8. Backend: ModelInferenceService Integration
  - Add `reasoning_service` to ModelInferenceService __init__
  - Update `generate_recommendations()` to accept include_reasoning and reasoning_level parameters
  - Implement `_extract_reasoning_trace()` method
  - Call model's `forward_with_reasoning_trace()` when reasoning enabled
  - Enhance recommendations with dynamic reasoning from ReasoningService
  - Handle reasoning levels (basic, detailed, full)
  - _Requirements: 3.1, 10.1, 10.5_

- [x] 8.1 Write unit tests for reasoning service integration
  - Test reasoning generation with various user profiles
  - Test reasoning level filtering (basic, detailed, full)
  - Test error handling when reasoning generation fails

- [x] 9. Backend: Error Handling and Performance
  - Add try-catch blocks for reasoning extraction failures
  - Implement fallback to basic reasoning on errors
  - Add logging for reasoning generation failures
  - Implement reasoning trace truncation for large traces
  - Add performance monitoring for reasoning generation time
  - _Requirements: 7.1, 7.4_

- [x] 10. API: Enhanced Recommendation Endpoint
  - Update `/api/recommendations` endpoint to accept include_reasoning query parameter
  - Add reasoning_level query parameter with validation (basic|detailed|full)
  - Update response model to EnhancedRecommendationResponse
  - Handle default behavior (basic reasoning when parameter not specified)
  - Add error handling for invalid reasoning_level values
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 10.1 Write integration tests for reasoning endpoint
  - Test include_reasoning=false (no reasoning)
  - Test include_reasoning=true (full reasoning)
  - Test default behavior (basic reasoning)
  - Test different reasoning levels
  - Test invalid parameter values

- [x] 11. Configuration and Feature Flags
  - Add REASONING_ENABLED environment variable
  - Add REASONING_DEFAULT_LEVEL environment variable
  - Add REASONING_MAX_THINKING_STEPS environment variable
  - Update config.py with reasoning configuration
  - Add feature flag checks in reasoning generation code
  - _Requirements: 7.5_

- [x] 12. OpenAPI Documentation
  - Update OpenAPI spec with new query parameters
  - Add schemas for all reasoning response models
  - Add examples for each reasoning level
  - Document error responses for reasoning failures
  - Add descriptions for all reasoning fields
  - _Requirements: 8.6_

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Performance Testing
  - Measure inference time without reasoning
  - Measure inference time with reasoning (basic, detailed, full)
  - Verify reasoning overhead is less than 10%
  - Test with concurrent requests
  - Test with large reasoning traces
  - _Requirements: 7.1_

- [x] 15. Integration Testing
  - Test full flow from API request to response with reasoning
  - Test reasoning with various user profiles and gifts
  - Test error scenarios (model failures, missing data)
  - Test caching behavior with reasoning
  - Test backward compatibility with old clients
