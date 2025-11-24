/**
 * Custom hook for managing reasoning panel state
 * Handles panel open/close, filter selections, and chart type preferences
 * Persists preferences to localStorage
 */

import { useState, useCallback, useEffect } from 'react';
import type { ReasoningFilter, ChartType } from '@/types/reasoning';

export interface UseReasoningPanelReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  activeFilters: ReasoningFilter[];
  setFilters: (filters: ReasoningFilter[]) => void;
  chartType: ChartType;
  setChartType: (type: ChartType) => void;
}

const STORAGE_KEY_FILTERS = 'reasoning-panel-filters';
const STORAGE_KEY_CHART_TYPE = 'reasoning-panel-chart-type';

/**
 * Hook for managing reasoning panel state
 * 
 * @returns Panel state and control functions
 * 
 * @example
 * ```tsx
 * const panel = useReasoningPanel();
 * 
 * <Button onClick={panel.open}>Show Details</Button>
 * <ReasoningPanel 
 *   isOpen={panel.isOpen}
 *   onClose={panel.close}
 *   activeFilters={panel.activeFilters}
 *   chartType={panel.chartType}
 * />
 * ```
 */
export function useReasoningPanel(): UseReasoningPanelReturn {
  const [isOpen, setIsOpen] = useState(false);
  
  // Load filters from localStorage on mount
  const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_FILTERS);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      }
    } catch (error) {
      console.warn('Failed to load reasoning panel filters from localStorage:', error);
    }
    // Default: show all sections
    return ['tool_selection', 'category_matching', 'attention_weights', 'thinking_steps'];
  });

  // Load chart type from localStorage on mount
  const [chartType, setChartTypeState] = useState<ChartType>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_CHART_TYPE);
      if (stored === 'bar' || stored === 'radar') {
        return stored;
      }
    } catch (error) {
      console.warn('Failed to load chart type from localStorage:', error);
    }
    return 'bar'; // Default chart type
  });

  // Persist filters to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_FILTERS, JSON.stringify(activeFilters));
    } catch (error) {
      console.warn('Failed to save reasoning panel filters to localStorage:', error);
    }
  }, [activeFilters]);

  // Persist chart type to localStorage when it changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_CHART_TYPE, chartType);
    } catch (error) {
      console.warn('Failed to save chart type to localStorage:', error);
    }
  }, [chartType]);

  const open = useCallback(() => {
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  const setFilters = useCallback((filters: ReasoningFilter[]) => {
    setActiveFilters(filters);
  }, []);

  const setChartType = useCallback((type: ChartType) => {
    setChartTypeState(type);
  }, []);

  return {
    isOpen,
    open,
    close,
    toggle,
    activeFilters,
    setFilters,
    chartType,
    setChartType,
  };
}
