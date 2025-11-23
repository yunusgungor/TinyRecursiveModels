import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import fc from 'fast-check';
import { ThemeProvider, useTheme } from '../ThemeContext';
import { useAppStore } from '@/store/useAppStore';

/**
 * Feature: trendyol-gift-recommendation-web, Property 22: Theme Consistency Across Components
 */

describe('Property Test: Theme Consistency Across Components', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    // Reset zustand store
    useAppStore.setState({ theme: 'light' });
  });

  afterEach(() => {
    // Clean up DOM
    document.documentElement.classList.remove('dark');
  });

  it('should maintain theme consistency across multiple theme changes', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom('light' as const, 'dark' as const), { minLength: 1, maxLength: 20 }),
        (themeSequence) => {
          const wrapper = ({ children }: { children: React.ReactNode }) => (
            <ThemeProvider>{children}</ThemeProvider>
          );

          const { result } = renderHook(() => useTheme(), { wrapper });

          // Apply each theme in sequence
          themeSequence.forEach((theme) => {
            act(() => {
              result.current.setTheme(theme);
            });

            // Verify theme is applied consistently
            expect(result.current.theme).toBe(theme);
            
            // Verify DOM reflects the theme
            if (theme === 'dark') {
              expect(document.documentElement.classList.contains('dark')).toBe(true);
            } else {
              expect(document.documentElement.classList.contains('dark')).toBe(false);
            }

            // Verify store is updated
            expect(useAppStore.getState().theme).toBe(theme);
          });

          // Final state should match last theme in sequence
          const lastTheme = themeSequence[themeSequence.length - 1];
          expect(result.current.theme).toBe(lastTheme);
          expect(useAppStore.getState().theme).toBe(lastTheme);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should toggle theme consistently', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('light' as const, 'dark' as const),
        fc.integer({ min: 1, max: 10 }),
        (initialTheme, toggleCount) => {
          const wrapper = ({ children }: { children: React.ReactNode }) => (
            <ThemeProvider>{children}</ThemeProvider>
          );

          const { result } = renderHook(() => useTheme(), { wrapper });

          // Set initial theme
          act(() => {
            result.current.setTheme(initialTheme);
          });

          // Toggle multiple times
          for (let i = 0; i < toggleCount; i++) {
            act(() => {
              result.current.toggleTheme();
            });
          }

          // Calculate expected theme after toggles
          const expectedTheme = toggleCount % 2 === 0 ? initialTheme : (initialTheme === 'light' ? 'dark' : 'light');

          // Verify final state
          expect(result.current.theme).toBe(expectedTheme);
          expect(useAppStore.getState().theme).toBe(expectedTheme);
          
          // Verify DOM
          if (expectedTheme === 'dark') {
            expect(document.documentElement.classList.contains('dark')).toBe(true);
          } else {
            expect(document.documentElement.classList.contains('dark')).toBe(false);
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should persist theme across provider remounts', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('light' as const, 'dark' as const),
        (theme) => {
          const wrapper = ({ children }: { children: React.ReactNode }) => (
            <ThemeProvider>{children}</ThemeProvider>
          );

          // First render
          const { result: result1, unmount } = renderHook(() => useTheme(), { wrapper });

          act(() => {
            result1.current.setTheme(theme);
          });

          expect(result1.current.theme).toBe(theme);

          // Unmount
          unmount();

          // Second render (simulating remount)
          const { result: result2 } = renderHook(() => useTheme(), { wrapper });

          // Theme should persist from store
          expect(result2.current.theme).toBe(theme);
          expect(useAppStore.getState().theme).toBe(theme);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should apply theme class to document root for any theme value', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('light' as const, 'dark' as const),
        (theme) => {
          const wrapper = ({ children }: { children: React.ReactNode }) => (
            <ThemeProvider>{children}</ThemeProvider>
          );

          const { result } = renderHook(() => useTheme(), { wrapper });

          act(() => {
            result.current.setTheme(theme);
          });

          // Verify DOM class matches theme
          const hasDarkClass = document.documentElement.classList.contains('dark');
          
          if (theme === 'dark') {
            expect(hasDarkClass).toBe(true);
          } else {
            expect(hasDarkClass).toBe(false);
          }

          // Verify only one theme class is present (no duplicates)
          const darkClassCount = Array.from(document.documentElement.classList).filter(
            (cls) => cls === 'dark'
          ).length;
          
          if (theme === 'dark') {
            expect(darkClassCount).toBe(1);
          } else {
            expect(darkClassCount).toBe(0);
          }
        }
      ),
      { numRuns: 100 }
    );
  });
});
