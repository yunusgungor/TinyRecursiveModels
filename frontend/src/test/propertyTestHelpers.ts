/**
 * Property-based testing helpers using fast-check
 * Provides generators and utilities for reasoning data models
 */

import * as fc from 'fast-check';
import type {
  ToolSelectionReasoning,
  CategoryMatchingReasoning,
  AttentionWeights,
  ThinkingStep,
  ConfidenceExplanation,
  ReasoningTrace,
  GiftItem,
  EnhancedGiftRecommendation,
  UserProfile,
} from '../types/reasoning';

/**
 * Minimum number of iterations for property-based tests
 * As specified in the design document
 */
export const MIN_PBT_ITERATIONS = 100;

/**
 * Generator for tool selection reasoning
 */
export const arbToolSelectionReasoning = (): fc.Arbitrary<ToolSelectionReasoning> => {
  return fc.record({
    name: fc.constantFrom('rating_tool', 'trend_tool', 'availability_tool', 'price_tool'),
    selected: fc.boolean(),
    score: fc.double({ min: 0, max: 1 }),
    reason: fc.string({ minLength: 10, maxLength: 100 }),
    confidence: fc.double({ min: 0, max: 1 }),
    priority: fc.integer({ min: 1, max: 10 }),
    factors: fc.option(fc.dictionary(fc.string(), fc.double({ min: 0, max: 1 }))),
  });
};

/**
 * Generator for category matching reasoning
 */
export const arbCategoryMatchingReasoning = (): fc.Arbitrary<CategoryMatchingReasoning> => {
  return fc.record({
    category_name: fc.constantFrom('Electronics', 'Books', 'Clothing', 'Home & Garden', 'Sports'),
    score: fc.double({ min: 0, max: 1 }),
    reasons: fc.array(fc.string({ minLength: 10, maxLength: 50 }), { minLength: 1, maxLength: 5 }),
    feature_contributions: fc.dictionary(fc.string(), fc.double({ min: 0, max: 1 })),
  });
};

/**
 * Generator for attention weights
 */
export const arbAttentionWeights = (): fc.Arbitrary<AttentionWeights> => {
  return fc.record({
    user_features: fc.dictionary(
      fc.constantFrom('hobbies', 'budget', 'age', 'occasion'),
      fc.double({ min: 0, max: 1 })
    ),
    gift_features: fc.dictionary(
      fc.constantFrom('category', 'price', 'rating'),
      fc.double({ min: 0, max: 1 })
    ),
  });
};

/**
 * Generator for thinking steps
 */
export const arbThinkingStep = (): fc.Arbitrary<ThinkingStep> => {
  return fc.record({
    step: fc.integer({ min: 1, max: 20 }),
    action: fc.string({ minLength: 10, maxLength: 50 }),
    result: fc.string({ minLength: 10, maxLength: 100 }),
    insight: fc.string({ minLength: 10, maxLength: 100 }),
  });
};

/**
 * Generator for confidence explanation
 */
export const arbConfidenceExplanation = (): fc.Arbitrary<ConfidenceExplanation> => {
  return fc.record({
    score: fc.double({ min: 0, max: 1 }),
    level: fc.constantFrom('high', 'medium', 'low'),
    factors: fc.record({
      positive: fc.array(fc.string({ minLength: 10, maxLength: 50 }), { minLength: 0, maxLength: 5 }),
      negative: fc.array(fc.string({ minLength: 10, maxLength: 50 }), { minLength: 0, maxLength: 5 }),
    }),
  });
};

/**
 * Generator for reasoning trace
 */
export const arbReasoningTrace = (): fc.Arbitrary<ReasoningTrace> => {
  return fc.record({
    tool_selection: fc.array(arbToolSelectionReasoning(), { minLength: 1, maxLength: 5 }),
    category_matching: fc.array(arbCategoryMatchingReasoning(), { minLength: 3, maxLength: 10 }),
    attention_weights: arbAttentionWeights(),
    thinking_steps: fc.array(arbThinkingStep(), { minLength: 1, maxLength: 20 }),
    confidence_explanation: fc.option(arbConfidenceExplanation()),
  });
};

/**
 * Generator for gift item
 */
export const arbGiftItem = (): fc.Arbitrary<GiftItem> => {
  return fc.record({
    id: fc.uuid(),
    name: fc.string({ minLength: 5, maxLength: 50 }),
    price: fc.double({ min: 10, max: 10000 }),
    image_url: fc.option(fc.webUrl()),
    category: fc.constantFrom('Electronics', 'Books', 'Clothing', 'Home & Garden', 'Sports'),
    rating: fc.option(fc.double({ min: 0, max: 5 })),
    availability: fc.option(fc.boolean()),
    description: fc.option(fc.string({ minLength: 20, maxLength: 200 })),
  });
};

/**
 * Generator for enhanced gift recommendation
 */
export const arbEnhancedGiftRecommendation = (): fc.Arbitrary<EnhancedGiftRecommendation> => {
  return fc.record({
    gift: arbGiftItem(),
    reasoning: fc.array(fc.string({ minLength: 20, maxLength: 100 }), { minLength: 1, maxLength: 5 }),
    confidence: fc.double({ min: 0, max: 1 }),
    reasoning_trace: fc.option(arbReasoningTrace()),
  });
};

/**
 * Generator for user profile
 */
export const arbUserProfile = (): fc.Arbitrary<UserProfile> => {
  return fc.record({
    hobbies: fc.option(fc.array(fc.string({ minLength: 5, maxLength: 20 }), { minLength: 1, maxLength: 5 })),
    age: fc.option(fc.integer({ min: 18, max: 100 })),
    budget: fc.option(fc.double({ min: 50, max: 5000 })),
    occasion: fc.option(fc.constantFrom('birthday', 'anniversary', 'wedding', 'graduation', 'christmas')),
    gender: fc.option(fc.constantFrom('male', 'female', 'other')),
    relationship: fc.option(fc.constantFrom('friend', 'family', 'partner', 'colleague')),
  });
};

/**
 * Generator for confidence scores in specific ranges
 */
export const arbHighConfidence = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.8, max: 1.0 });
};

export const arbMediumConfidence = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.5, max: 0.8 });
};

export const arbLowConfidence = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.0, max: 0.5 });
};

/**
 * Generator for category scores in specific ranges
 */
export const arbHighCategoryScore = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.7, max: 1.0 });
};

export const arbLowCategoryScore = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.0, max: 0.3 });
};

export const arbMediumCategoryScore = (): fc.Arbitrary<number> => {
  return fc.double({ min: 0.3, max: 0.7 });
};

/**
 * Helper to run property tests with minimum iterations
 */
export const runPropertyTest = <T>(
  arbitrary: fc.Arbitrary<T>,
  predicate: (value: T) => boolean | void,
  options?: fc.Parameters<[T]>
): void => {
  fc.assert(
    fc.property(arbitrary, predicate),
    {
      numRuns: MIN_PBT_ITERATIONS,
      ...options,
    }
  );
};
