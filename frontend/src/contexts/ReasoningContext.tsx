/**
 * ReasoningContext - Centralized state management for reasoning features
 * 
 * This context provides:
 * - Reasoning level management (basic/detailed/full)
 * - Gift selection for comparison mode
 * - Comparison mode state
 * - Reasoning panel state (open/close, filters, chart type)
 * - localStorage persistence for user preferences
 * 
 * Requirements: 7.5, 7.6, 12.1
 */

import { createContext, useContext, ReactNode } from 'react';
import { useReasoningLevel } from '@/hooks/useReasoningLevel';
import { useReasoningPanel } from '@/hooks/useReasoningPanel';
import { useAppStore } from '@/store/useAppStore';
import type { ReasoningLevel, ReasoningFilter, ChartType } from '@/types/reasoning';

interface ReasoningContextValue {
  // Reasoning level management
  reasoningLevel: ReasoningLevel;
  setReasoningLevel: (level: ReasoningLevel) => void;
  
  // Gift selection for comparison
  selectedGifts: string[];
  toggleGiftSelection: (giftId: string) => void;
  clearSelection: () => void;
  
  // Comparison mode
  isComparisonMode: boolean;
  setComparisonMode: (enabled: boolean) => void;
  
  // Reasoning panel state
  isPanelOpen: boolean;
  openPanel: () => void;
  closePanel: () => void;
  togglePanel: () => void;
  
  // Panel filters and preferences
  activeFilters: ReasoningFilter[];
  setFilters: (filters: ReasoningFilter[]) => void;
  chartType: ChartType;
  setChartType: (type: ChartType) => void;
}

const ReasoningContext = createContext<ReasoningContextValue | undefined>(undefined);

export interface ReasoningProviderProps {
  children: ReactNode;
}

/**
 * ReasoningProvider - Provides reasoning state to the component tree
 * 
 * Integrates:
 * - useReasoningLevel hook for reasoning level persistence
 * - useReasoningPanel hook for panel state and preferences
 * - useAppStore for comparison mode and gift selection
 * 
 * @example
 * ```tsx
 * <ReasoningProvider>
 *   <App />
 * </ReasoningProvider>
 * ```
 */
export function ReasoningProvider({ children }: ReasoningProviderProps) {
  // Reasoning level management (with localStorage persistence)
  const { level, setLevel } = useReasoningLevel();
  
  // Reasoning panel state (with localStorage persistence)
  const {
    isOpen,
    open,
    close,
    toggle,
    activeFilters,
    setFilters,
    chartType,
    setChartType,
  } = useReasoningPanel();
  
  // Comparison mode and gift selection (from Zustand store)
  const {
    selectedGiftsForComparison,
    isComparisonMode,
    toggleGiftSelection: toggleGiftInStore,
    clearGiftSelection,
    setComparisonMode: setComparisonModeInStore,
  } = useAppStore();

  const value: ReasoningContextValue = {
    // Reasoning level
    reasoningLevel: level,
    setReasoningLevel: setLevel,
    
    // Gift selection
    selectedGifts: selectedGiftsForComparison,
    toggleGiftSelection: toggleGiftInStore,
    clearSelection: clearGiftSelection,
    
    // Comparison mode
    isComparisonMode,
    setComparisonMode: setComparisonModeInStore,
    
    // Panel state
    isPanelOpen: isOpen,
    openPanel: open,
    closePanel: close,
    togglePanel: toggle,
    
    // Panel preferences
    activeFilters,
    setFilters,
    chartType,
    setChartType,
  };

  return (
    <ReasoningContext.Provider value={value}>
      {children}
    </ReasoningContext.Provider>
  );
}

/**
 * useReasoningContext - Hook to access reasoning context
 * 
 * @throws Error if used outside ReasoningProvider
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { reasoningLevel, setReasoningLevel, openPanel } = useReasoningContext();
 *   
 *   return (
 *     <div>
 *       <Select value={reasoningLevel} onValueChange={setReasoningLevel}>
 *         <SelectItem value="basic">Basic</SelectItem>
 *         <SelectItem value="detailed">Detailed</SelectItem>
 *         <SelectItem value="full">Full</SelectItem>
 *       </Select>
 *       <Button onClick={openPanel}>Show Details</Button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useReasoningContext(): ReasoningContextValue {
  const context = useContext(ReasoningContext);
  if (context === undefined) {
    throw new Error('useReasoningContext must be used within a ReasoningProvider');
  }
  return context;
}
