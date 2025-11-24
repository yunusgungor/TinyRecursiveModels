/**
 * Custom hook for responsive design using media queries
 * Provides real-time updates when viewport size changes
 */

import { useState, useEffect } from 'react';

/**
 * Hook for detecting media query matches
 * 
 * @param query - CSS media query string
 * @returns Boolean indicating if the media query matches
 * 
 * @example
 * ```tsx
 * const isMobile = useMediaQuery('(max-width: 767px)');
 * const isDesktop = useMediaQuery('(min-width: 1024px)');
 * 
 * return (
 *   <div>
 *     {isMobile ? <MobileLayout /> : <DesktopLayout />}
 *   </div>
 * );
 * ```
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    // Check if window is available (SSR safety)
    if (typeof window !== 'undefined') {
      return window.matchMedia(query).matches;
    }
    return false;
  });

  useEffect(() => {
    // Check if window is available (SSR safety)
    if (typeof window === 'undefined') {
      return;
    }

    const media = window.matchMedia(query);
    
    // Set initial value
    setMatches(media.matches);

    // Create event listener
    const listener = (event: MediaQueryListEvent) => {
      setMatches(event.matches);
    };

    // Add listener (modern browsers)
    if (media.addEventListener) {
      media.addEventListener('change', listener);
    } else {
      // Fallback for older browsers
      media.addListener(listener);
    }

    // Cleanup
    return () => {
      if (media.removeEventListener) {
        media.removeEventListener('change', listener);
      } else {
        // Fallback for older browsers
        media.removeListener(listener);
      }
    };
  }, [query]);

  return matches;
}

/**
 * Predefined breakpoints for common screen sizes
 */
export const breakpoints = {
  mobile: '(max-width: 767px)',
  tablet: '(min-width: 768px) and (max-width: 1023px)',
  desktop: '(min-width: 1024px)',
  sm: '(min-width: 640px)',
  md: '(min-width: 768px)',
  lg: '(min-width: 1024px)',
  xl: '(min-width: 1280px)',
  '2xl': '(min-width: 1536px)',
} as const;

/**
 * Hook for detecting mobile viewport
 */
export function useIsMobile(): boolean {
  return useMediaQuery(breakpoints.mobile);
}

/**
 * Hook for detecting tablet viewport
 */
export function useIsTablet(): boolean {
  return useMediaQuery(breakpoints.tablet);
}

/**
 * Hook for detecting desktop viewport
 */
export function useIsDesktop(): boolean {
  return useMediaQuery(breakpoints.desktop);
}
