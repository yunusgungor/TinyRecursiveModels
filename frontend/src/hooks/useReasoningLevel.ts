/**
 * Custom hook for managing reasoning level preference
 * Persists to localStorage and loads on mount
 */

import { useState, useEffect, useCallback } from 'react';
import type { ReasoningLevel } from '@/types/reasoning';

export interface UseReasoningLevelReturn {
  level: ReasoningLevel;
  setLevel: (level: ReasoningLevel) => void;
}

const STORAGE_KEY = 'reasoning-level';
const DEFAULT_LEVEL: ReasoningLevel = 'detailed';

/**
 * Hook for managing reasoning level preference with localStorage persistence
 * 
 * @returns Current reasoning level and setter function
 * 
 * @example
 * ```tsx
 * const { level, setLevel } = useReasoningLevel();
 * 
 * <Select value={level} onValueChange={setLevel}>
 *   <SelectItem value="basic">Basic</SelectItem>
 *   <SelectItem value="detailed">Detailed</SelectItem>
 *   <SelectItem value="full">Full</SelectItem>
 * </Select>
 * ```
 */
export function useReasoningLevel(): UseReasoningLevelReturn {
  // Load from localStorage on mount
  const [level, setLevelState] = useState<ReasoningLevel>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'basic' || stored === 'detailed' || stored === 'full') {
        return stored;
      }
    } catch (error) {
      console.warn('Failed to load reasoning level from localStorage:', error);
    }
    return DEFAULT_LEVEL;
  });

  // Persist to localStorage when level changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, level);
    } catch (error) {
      console.warn('Failed to save reasoning level to localStorage:', error);
    }
  }, [level]);

  const setLevel = useCallback((newLevel: ReasoningLevel) => {
    setLevelState(newLevel);
  }, []);

  return {
    level,
    setLevel,
  };
}
