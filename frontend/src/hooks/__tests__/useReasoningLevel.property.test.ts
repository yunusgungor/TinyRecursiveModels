/**
 * Property-based tests for useReasoningLevel hook
 * Feature: frontend-reasoning-visualization, Property 35: Reasoning level persistence (round-trip)
 * Validates: Requirements 7.5, 7.6
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as fc from 'fast-check';
import { useReasoningLevel } from '../useReasoningLevel';
import type { ReasoningLevel } from '@/types/reasoning';
import { MIN_PBT_ITERATIONS } from '@/test/propertyTestHelpers';

describe('useReasoningLevel - Property-Based Tests', () => {
  // Clear localStorage before and after each test
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  /**
   * Property 35: Reasoning level persistence (round-trip)
   * For any reasoning level, saving it to localStorage and reloading should return the same level
   */
  it('Property 35: should persist reasoning level through localStorage round-trip', () => {
    // Generator for valid reasoning levels
    const arbReasoningLevel = fc.constantFrom<ReasoningLevel>('basic', 'detailed', 'full');

    fc.assert(
      fc.property(arbReasoningLevel, (level) => {
        // Step 1: Set the reasoning level
        const { result: result1 } = renderHook(() => useReasoningLevel());
        
        act(() => {
          result1.current.setLevel(level);
        });

        // Verify it was set correctly
        expect(result1.current.level).toBe(level);

        // Step 2: Simulate page reload by creating a new hook instance
        // This should load the value from localStorage
        const { result: result2 } = renderHook(() => useReasoningLevel());

        // Step 3: Verify the round-trip - the loaded value should match the saved value
        expect(result2.current.level).toBe(level);

        // Additional verification: check localStorage directly
        const storedValue = localStorage.getItem('reasoning-level');
        expect(storedValue).toBe(level);
      }),
      { numRuns: MIN_PBT_ITERATIONS }
    );
  });

  /**
   * Property: Multiple updates should preserve the last value
   * For any sequence of reasoning level changes, only the last one should be persisted
   */
  it('should preserve only the last reasoning level after multiple updates', () => {
    const arbReasoningLevel = fc.constantFrom<ReasoningLevel>('basic', 'detailed', 'full');
    const arbReasoningLevelArray = fc.array(arbReasoningLevel, { minLength: 1, maxLength: 10 });

    fc.assert(
      fc.property(arbReasoningLevelArray, (levels) => {
        const { result } = renderHook(() => useReasoningLevel());

        // Apply all level changes
        act(() => {
          levels.forEach(level => {
            result.current.setLevel(level);
          });
        });

        // The current level should be the last one in the array
        const lastLevel = levels[levels.length - 1];
        expect(result.current.level).toBe(lastLevel);

        // Verify localStorage has the last value
        const storedValue = localStorage.getItem('reasoning-level');
        expect(storedValue).toBe(lastLevel);

        // Verify round-trip with new hook instance
        const { result: result2 } = renderHook(() => useReasoningLevel());
        expect(result2.current.level).toBe(lastLevel);
      }),
      { numRuns: MIN_PBT_ITERATIONS }
    );
  });

  /**
   * Property: Default value should be 'detailed' when localStorage is empty
   */
  it('should use default value when localStorage is empty', () => {
    // Ensure localStorage is empty
    localStorage.clear();

    const { result } = renderHook(() => useReasoningLevel());

    // Should default to 'detailed'
    expect(result.current.level).toBe('detailed');
  });

  /**
   * Property: Should handle corrupted localStorage gracefully
   */
  it('should handle corrupted localStorage data gracefully', () => {
    const arbInvalidValue = fc.oneof(
      fc.constant('invalid'),
      fc.constant('{}'),
      fc.constant('[]'),
      fc.constant('123'),
      fc.constant('null'),
      fc.constant('undefined')
    );

    fc.assert(
      fc.property(arbInvalidValue, (invalidValue) => {
        // Set invalid value in localStorage
        localStorage.setItem('reasoning-level', invalidValue);

        // Hook should still work and use default value
        const { result } = renderHook(() => useReasoningLevel());

        // Should fall back to default 'detailed'
        expect(result.current.level).toBe('detailed');

        // Should be able to set a valid value
        act(() => {
          result.current.setLevel('basic');
        });

        expect(result.current.level).toBe('basic');
      }),
      { numRuns: MIN_PBT_ITERATIONS }
    );
  });

  /**
   * Property: Idempotence - setting the same value multiple times should work correctly
   */
  it('should handle setting the same value multiple times (idempotence)', () => {
    const arbReasoningLevel = fc.constantFrom<ReasoningLevel>('basic', 'detailed', 'full');
    const arbRepeatCount = fc.integer({ min: 1, max: 10 });

    fc.assert(
      fc.property(arbReasoningLevel, arbRepeatCount, (level, repeatCount) => {
        const { result } = renderHook(() => useReasoningLevel());

        // Set the same value multiple times
        act(() => {
          for (let i = 0; i < repeatCount; i++) {
            result.current.setLevel(level);
          }
        });

        // Should still have the correct value
        expect(result.current.level).toBe(level);

        // Verify localStorage
        const storedValue = localStorage.getItem('reasoning-level');
        expect(storedValue).toBe(level);

        // Verify round-trip
        const { result: result2 } = renderHook(() => useReasoningLevel());
        expect(result2.current.level).toBe(level);
      }),
      { numRuns: MIN_PBT_ITERATIONS }
    );
  });
});
