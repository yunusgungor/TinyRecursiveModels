/**
 * Unit tests for useReasoningPanel hook
 * Tests state management, filter handling, and localStorage persistence
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useReasoningPanel } from '../useReasoningPanel';
import type { ReasoningFilter } from '@/types/reasoning';

describe('useReasoningPanel', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(() => useReasoningPanel());

    expect(result.current.isOpen).toBe(false);
    expect(result.current.activeFilters).toEqual([
      'tool_selection',
      'category_matching',
      'attention_weights',
      'thinking_steps',
    ]);
    expect(result.current.chartType).toBe('bar');
  });

  it('should open panel', () => {
    const { result } = renderHook(() => useReasoningPanel());

    act(() => {
      result.current.open();
    });

    expect(result.current.isOpen).toBe(true);
  });

  it('should close panel', () => {
    const { result } = renderHook(() => useReasoningPanel());

    act(() => {
      result.current.open();
    });

    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.close();
    });

    expect(result.current.isOpen).toBe(false);
  });

  it('should toggle panel state', () => {
    const { result } = renderHook(() => useReasoningPanel());

    expect(result.current.isOpen).toBe(false);

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isOpen).toBe(false);
  });

  it('should update active filters', () => {
    const { result } = renderHook(() => useReasoningPanel());

    const newFilters: ReasoningFilter[] = ['tool_selection', 'category_matching'];

    act(() => {
      result.current.setFilters(newFilters);
    });

    expect(result.current.activeFilters).toEqual(newFilters);
  });

  it('should update chart type', () => {
    const { result } = renderHook(() => useReasoningPanel());

    expect(result.current.chartType).toBe('bar');

    act(() => {
      result.current.setChartType('radar');
    });

    expect(result.current.chartType).toBe('radar');
  });

  it('should persist filters to localStorage', () => {
    const { result } = renderHook(() => useReasoningPanel());

    const newFilters: ReasoningFilter[] = ['tool_selection'];

    act(() => {
      result.current.setFilters(newFilters);
    });

    const stored = localStorage.getItem('reasoning-panel-filters');
    expect(stored).toBe(JSON.stringify(newFilters));
  });

  it('should persist chart type to localStorage', () => {
    const { result } = renderHook(() => useReasoningPanel());

    act(() => {
      result.current.setChartType('radar');
    });

    const stored = localStorage.getItem('reasoning-panel-chart-type');
    expect(stored).toBe('radar');
  });

  it('should load filters from localStorage on mount', () => {
    const savedFilters: ReasoningFilter[] = ['category_matching', 'attention_weights'];
    localStorage.setItem('reasoning-panel-filters', JSON.stringify(savedFilters));

    const { result } = renderHook(() => useReasoningPanel());

    expect(result.current.activeFilters).toEqual(savedFilters);
  });

  it('should load chart type from localStorage on mount', () => {
    localStorage.setItem('reasoning-panel-chart-type', 'radar');

    const { result } = renderHook(() => useReasoningPanel());

    expect(result.current.chartType).toBe('radar');
  });

  it('should handle corrupted filters in localStorage', () => {
    localStorage.setItem('reasoning-panel-filters', 'invalid json');

    const { result } = renderHook(() => useReasoningPanel());

    // Should fall back to default filters
    expect(result.current.activeFilters).toEqual([
      'tool_selection',
      'category_matching',
      'attention_weights',
      'thinking_steps',
    ]);
  });

  it('should handle invalid chart type in localStorage', () => {
    localStorage.setItem('reasoning-panel-chart-type', 'invalid');

    const { result } = renderHook(() => useReasoningPanel());

    // Should fall back to default chart type
    expect(result.current.chartType).toBe('bar');
  });

  it('should handle non-array filters in localStorage', () => {
    localStorage.setItem('reasoning-panel-filters', JSON.stringify({ invalid: 'object' }));

    const { result } = renderHook(() => useReasoningPanel());

    // Should fall back to default filters
    expect(result.current.activeFilters).toEqual([
      'tool_selection',
      'category_matching',
      'attention_weights',
      'thinking_steps',
    ]);
  });

  it('should allow empty filters array', () => {
    const { result } = renderHook(() => useReasoningPanel());

    act(() => {
      result.current.setFilters([]);
    });

    expect(result.current.activeFilters).toEqual([]);

    // Should persist empty array
    const stored = localStorage.getItem('reasoning-panel-filters');
    expect(stored).toBe('[]');
  });

  it('should maintain state independence between multiple instances', () => {
    const { result: result1 } = renderHook(() => useReasoningPanel());
    const { result: result2 } = renderHook(() => useReasoningPanel());

    // Both should start with same state from localStorage
    expect(result1.current.activeFilters).toEqual(result2.current.activeFilters);

    // Changing one should affect the other through localStorage
    act(() => {
      result1.current.setFilters(['tool_selection']);
    });

    // result2 won't automatically update, but a new instance will have the updated value
    const { result: result3 } = renderHook(() => useReasoningPanel());
    expect(result3.current.activeFilters).toEqual(['tool_selection']);
  });
});
