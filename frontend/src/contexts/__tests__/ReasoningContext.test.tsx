/**
 * Unit tests for ReasoningContext
 * 
 * Tests:
 * - Context provider functionality
 * - State updates for reasoning level
 * - State updates for gift selection
 * - State updates for comparison mode
 * - State updates for panel state
 * - State updates for filters and chart type
 * - localStorage persistence
 * - Error handling when used outside provider
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { ReasoningProvider, useReasoningContext } from '../ReasoningContext';
import { useAppStore } from '@/store/useAppStore';

describe('ReasoningContext', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    
    // Reset app store to initial state
    useAppStore.setState({
      selectedGiftsForComparison: [],
      isComparisonMode: false,
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('Context Provider', () => {
    it('should provide reasoning context to children', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current).toBeDefined();
      expect(result.current.reasoningLevel).toBeDefined();
      expect(result.current.setReasoningLevel).toBeDefined();
      expect(result.current.selectedGifts).toBeDefined();
      expect(result.current.toggleGiftSelection).toBeDefined();
      expect(result.current.isComparisonMode).toBeDefined();
      expect(result.current.setComparisonMode).toBeDefined();
      expect(result.current.isPanelOpen).toBeDefined();
      expect(result.current.openPanel).toBeDefined();
      expect(result.current.closePanel).toBeDefined();
      expect(result.current.togglePanel).toBeDefined();
      expect(result.current.activeFilters).toBeDefined();
      expect(result.current.setFilters).toBeDefined();
      expect(result.current.chartType).toBeDefined();
      expect(result.current.setChartType).toBeDefined();
    });

    it('should throw error when used outside provider', () => {
      expect(() => {
        renderHook(() => useReasoningContext());
      }).toThrow('useReasoningContext must be used within a ReasoningProvider');
    });
  });

  describe('Reasoning Level Management', () => {
    it('should have default reasoning level as detailed', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.reasoningLevel).toBe('detailed');
    });

    it('should update reasoning level', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setReasoningLevel('full');
      });

      expect(result.current.reasoningLevel).toBe('full');
    });

    it('should persist reasoning level to localStorage', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setReasoningLevel('basic');
      });

      expect(localStorage.getItem('reasoning-level')).toBe('basic');
    });

    it('should load reasoning level from localStorage on mount', () => {
      localStorage.setItem('reasoning-level', 'full');

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.reasoningLevel).toBe('full');
    });

    it('should update reasoning level multiple times', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setReasoningLevel('basic');
      });
      expect(result.current.reasoningLevel).toBe('basic');

      act(() => {
        result.current.setReasoningLevel('full');
      });
      expect(result.current.reasoningLevel).toBe('full');

      act(() => {
        result.current.setReasoningLevel('detailed');
      });
      expect(result.current.reasoningLevel).toBe('detailed');
    });
  });

  describe('Gift Selection Management', () => {
    it('should start with empty gift selection', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.selectedGifts).toEqual([]);
    });

    it('should toggle gift selection', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
      });

      expect(result.current.selectedGifts).toContain('gift-1');
    });

    it('should toggle gift selection off', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
      });
      expect(result.current.selectedGifts).toContain('gift-1');

      act(() => {
        result.current.toggleGiftSelection('gift-1');
      });
      expect(result.current.selectedGifts).not.toContain('gift-1');
    });

    it('should select multiple gifts', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
        result.current.toggleGiftSelection('gift-2');
        result.current.toggleGiftSelection('gift-3');
      });

      expect(result.current.selectedGifts).toEqual(['gift-1', 'gift-2', 'gift-3']);
    });

    it('should clear gift selection', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
        result.current.toggleGiftSelection('gift-2');
      });
      expect(result.current.selectedGifts.length).toBe(2);

      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.selectedGifts).toEqual([]);
    });
  });

  describe('Comparison Mode Management', () => {
    it('should start with comparison mode disabled', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.isComparisonMode).toBe(false);
    });

    it('should enable comparison mode', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setComparisonMode(true);
      });

      expect(result.current.isComparisonMode).toBe(true);
    });

    it('should disable comparison mode', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setComparisonMode(true);
      });
      expect(result.current.isComparisonMode).toBe(true);

      act(() => {
        result.current.setComparisonMode(false);
      });

      expect(result.current.isComparisonMode).toBe(false);
    });

    it('should clear gift selection when disabling comparison mode', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
        result.current.toggleGiftSelection('gift-2');
        result.current.setComparisonMode(true);
      });
      expect(result.current.selectedGifts.length).toBe(2);

      act(() => {
        result.current.setComparisonMode(false);
      });

      expect(result.current.selectedGifts).toEqual([]);
    });

    it('should clear both comparison mode and selection when calling clearSelection', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.toggleGiftSelection('gift-1');
        result.current.setComparisonMode(true);
      });
      expect(result.current.isComparisonMode).toBe(true);
      expect(result.current.selectedGifts.length).toBe(1);

      act(() => {
        result.current.clearSelection();
      });

      expect(result.current.isComparisonMode).toBe(false);
      expect(result.current.selectedGifts).toEqual([]);
    });
  });

  describe('Panel State Management', () => {
    it('should start with panel closed', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.isPanelOpen).toBe(false);
    });

    it('should open panel', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.openPanel();
      });

      expect(result.current.isPanelOpen).toBe(true);
    });

    it('should close panel', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.openPanel();
      });
      expect(result.current.isPanelOpen).toBe(true);

      act(() => {
        result.current.closePanel();
      });

      expect(result.current.isPanelOpen).toBe(false);
    });

    it('should toggle panel', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.isPanelOpen).toBe(false);

      act(() => {
        result.current.togglePanel();
      });
      expect(result.current.isPanelOpen).toBe(true);

      act(() => {
        result.current.togglePanel();
      });
      expect(result.current.isPanelOpen).toBe(false);
    });
  });

  describe('Filter Management', () => {
    it('should have default filters showing all sections', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.activeFilters).toEqual([
        'tool_selection',
        'category_matching',
        'attention_weights',
        'thinking_steps',
      ]);
    });

    it('should update filters', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setFilters(['tool_selection', 'category_matching']);
      });

      expect(result.current.activeFilters).toEqual(['tool_selection', 'category_matching']);
    });

    it('should persist filters to localStorage', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setFilters(['attention_weights']);
      });

      const stored = localStorage.getItem('reasoning-panel-filters');
      expect(stored).toBeDefined();
      expect(JSON.parse(stored!)).toEqual(['attention_weights']);
    });

    it('should load filters from localStorage on mount', () => {
      localStorage.setItem(
        'reasoning-panel-filters',
        JSON.stringify(['tool_selection'])
      );

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.activeFilters).toEqual(['tool_selection']);
    });
  });

  describe('Chart Type Management', () => {
    it('should have default chart type as bar', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.chartType).toBe('bar');
    });

    it('should update chart type', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setChartType('radar');
      });

      expect(result.current.chartType).toBe('radar');
    });

    it('should persist chart type to localStorage', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setChartType('radar');
      });

      expect(localStorage.getItem('reasoning-panel-chart-type')).toBe('radar');
    });

    it('should load chart type from localStorage on mount', () => {
      localStorage.setItem('reasoning-panel-chart-type', 'radar');

      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.chartType).toBe('radar');
    });

    it('should toggle between chart types', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result.current.chartType).toBe('bar');

      act(() => {
        result.current.setChartType('radar');
      });
      expect(result.current.chartType).toBe('radar');

      act(() => {
        result.current.setChartType('bar');
      });
      expect(result.current.chartType).toBe('bar');
    });
  });

  describe('Integration Tests', () => {
    it('should handle multiple state updates correctly', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result.current.setReasoningLevel('full');
        result.current.toggleGiftSelection('gift-1');
        result.current.toggleGiftSelection('gift-2');
        result.current.setComparisonMode(true);
        result.current.openPanel();
        result.current.setFilters(['tool_selection', 'attention_weights']);
        result.current.setChartType('radar');
      });

      expect(result.current.reasoningLevel).toBe('full');
      expect(result.current.selectedGifts).toEqual(['gift-1', 'gift-2']);
      expect(result.current.isComparisonMode).toBe(true);
      expect(result.current.isPanelOpen).toBe(true);
      expect(result.current.activeFilters).toEqual(['tool_selection', 'attention_weights']);
      expect(result.current.chartType).toBe('radar');
    });

    it('should persist all preferences after remount', () => {
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <ReasoningProvider>{children}</ReasoningProvider>
      );

      const { result: result1, unmount } = renderHook(() => useReasoningContext(), { wrapper });

      act(() => {
        result1.current.setReasoningLevel('basic');
        result1.current.setFilters(['category_matching']);
        result1.current.setChartType('radar');
      });

      unmount();

      const { result: result2 } = renderHook(() => useReasoningContext(), { wrapper });

      expect(result2.current.reasoningLevel).toBe('basic');
      expect(result2.current.activeFilters).toEqual(['category_matching']);
      expect(result2.current.chartType).toBe('radar');
    });
  });
});
