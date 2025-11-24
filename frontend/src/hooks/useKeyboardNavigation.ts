/**
 * Custom hook for keyboard navigation accessibility
 * Provides arrow key navigation and Enter/Space selection
 */

import { useState, useCallback, useEffect, KeyboardEvent } from 'react';

export interface UseKeyboardNavigationOptions {
  /**
   * Number of items to navigate through
   */
  itemCount: number;
  
  /**
   * Callback when an item is selected (Enter or Space pressed)
   */
  onSelect?: (index: number) => void;
  
  /**
   * Initial focused index
   */
  initialIndex?: number;
  
  /**
   * Whether navigation should loop (wrap around)
   */
  loop?: boolean;
  
  /**
   * Whether the navigation is enabled
   */
  enabled?: boolean;
}

export interface UseKeyboardNavigationReturn {
  /**
   * Currently focused item index
   */
  focusedIndex: number;
  
  /**
   * Set the focused index manually
   */
  setFocusedIndex: (index: number) => void;
  
  /**
   * Handle keyboard events (attach to container element)
   */
  handleKeyDown: (event: KeyboardEvent) => void;
  
  /**
   * Get props for an item at a specific index
   */
  getItemProps: (index: number) => {
    tabIndex: number;
    'data-focused': boolean;
    onFocus: () => void;
  };
}

/**
 * Hook for keyboard navigation with arrow keys
 * 
 * @param options - Configuration options
 * @returns Navigation state and handlers
 * 
 * @example
 * ```tsx
 * const { focusedIndex, handleKeyDown, getItemProps } = useKeyboardNavigation({
 *   itemCount: items.length,
 *   onSelect: (index) => console.log('Selected:', items[index]),
 *   loop: true,
 * });
 * 
 * return (
 *   <div onKeyDown={handleKeyDown}>
 *     {items.map((item, index) => (
 *       <div key={index} {...getItemProps(index)}>
 *         {item.name}
 *       </div>
 *     ))}
 *   </div>
 * );
 * ```
 */
export function useKeyboardNavigation(
  options: UseKeyboardNavigationOptions
): UseKeyboardNavigationReturn {
  const {
    itemCount,
    onSelect,
    initialIndex = 0,
    loop = false,
    enabled = true,
  } = options;

  const [focusedIndex, setFocusedIndex] = useState(initialIndex);

  // Reset focused index if item count changes
  useEffect(() => {
    if (focusedIndex >= itemCount) {
      setFocusedIndex(Math.max(0, itemCount - 1));
    }
  }, [itemCount, focusedIndex]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled || itemCount === 0) {
        return;
      }

      switch (event.key) {
        case 'ArrowDown':
        case 'Down': // IE/Edge
          event.preventDefault();
          setFocusedIndex((prev) => {
            if (prev >= itemCount - 1) {
              return loop ? 0 : prev;
            }
            return prev + 1;
          });
          break;

        case 'ArrowUp':
        case 'Up': // IE/Edge
          event.preventDefault();
          setFocusedIndex((prev) => {
            if (prev <= 0) {
              return loop ? itemCount - 1 : prev;
            }
            return prev - 1;
          });
          break;

        case 'Home':
          event.preventDefault();
          setFocusedIndex(0);
          break;

        case 'End':
          event.preventDefault();
          setFocusedIndex(itemCount - 1);
          break;

        case 'Enter':
        case ' ':
        case 'Spacebar': // IE/Edge
          event.preventDefault();
          if (onSelect) {
            onSelect(focusedIndex);
          }
          break;

        default:
          // Allow other keys to propagate
          break;
      }
    },
    [enabled, itemCount, loop, onSelect, focusedIndex]
  );

  const getItemProps = useCallback(
    (index: number) => ({
      tabIndex: index === focusedIndex ? 0 : -1,
      'data-focused': index === focusedIndex,
      onFocus: () => setFocusedIndex(index),
    }),
    [focusedIndex]
  );

  return {
    focusedIndex,
    setFocusedIndex,
    handleKeyDown,
    getItemProps,
  };
}

/**
 * Hook for simple tab navigation (no arrow keys)
 * Useful for simpler navigation patterns
 */
export function useTabNavigation(itemCount: number) {
  const [focusedIndex, setFocusedIndex] = useState(0);

  const focusNext = useCallback(() => {
    setFocusedIndex((prev) => Math.min(prev + 1, itemCount - 1));
  }, [itemCount]);

  const focusPrevious = useCallback(() => {
    setFocusedIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const focusFirst = useCallback(() => {
    setFocusedIndex(0);
  }, []);

  const focusLast = useCallback(() => {
    setFocusedIndex(itemCount - 1);
  }, [itemCount]);

  return {
    focusedIndex,
    setFocusedIndex,
    focusNext,
    focusPrevious,
    focusFirst,
    focusLast,
  };
}
