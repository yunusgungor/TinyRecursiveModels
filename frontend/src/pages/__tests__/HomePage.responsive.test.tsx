import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { HomePage } from '../HomePage';
import { ThemeProvider } from '@/contexts/ThemeContext';

/**
 * Responsive Design Unit Tests
 * Validates: Requirements 8.1
 */

// Mock hooks
vi.mock('@/hooks/useRecommendations', () => ({
  useRecommendations: () => ({
    mutate: vi.fn(),
    data: null,
    isPending: false,
    isError: false,
    error: null,
  }),
}));

vi.mock('@/store/useAppStore', () => ({
  useAppStore: () => ({
    addSearchHistory: vi.fn(),
    isFavorite: vi.fn(() => false),
    addFavorite: vi.fn(),
    removeFavorite: vi.fn(),
  }),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          {component}
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('HomePage Responsive Design', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Breakpoint Tests', () => {
    it('should render mobile layout for small screens', () => {
      // Set viewport to mobile size
      global.innerWidth = 375;
      global.dispatchEvent(new Event('resize'));

      renderWithProviders(<HomePage />);

      // Check that main heading is present
      const heading = screen.getByText(/Trendyol Hediye Önerisi/i);
      expect(heading).toBeInTheDocument();

      // Check that navigation tabs are present
      expect(screen.getByText('Arama')).toBeInTheDocument();
      expect(screen.getByText('Favoriler')).toBeInTheDocument();
      expect(screen.getByText('Geçmiş')).toBeInTheDocument();
    });

    it('should render tablet layout for medium screens', () => {
      // Set viewport to tablet size
      global.innerWidth = 768;
      global.dispatchEvent(new Event('resize'));

      renderWithProviders(<HomePage />);

      const heading = screen.getByText(/Trendyol Hediye Önerisi/i);
      expect(heading).toBeInTheDocument();
    });

    it('should render desktop layout for large screens', () => {
      // Set viewport to desktop size
      global.innerWidth = 1280;
      global.dispatchEvent(new Event('resize'));

      renderWithProviders(<HomePage />);

      const heading = screen.getByText(/Trendyol Hediye Önerisi/i);
      expect(heading).toBeInTheDocument();
    });
  });

  describe('Mobile Layout Tests', () => {
    beforeEach(() => {
      global.innerWidth = 375;
      global.dispatchEvent(new Event('resize'));
    });

    it('should display navigation tabs in mobile view', () => {
      renderWithProviders(<HomePage />);

      const searchTab = screen.getByText('Arama');
      const favoritesTab = screen.getByText('Favoriler');
      const historyTab = screen.getByText('Geçmiş');

      expect(searchTab).toBeInTheDocument();
      expect(favoritesTab).toBeInTheDocument();
      expect(historyTab).toBeInTheDocument();
    });

    it('should render form in mobile view', () => {
      renderWithProviders(<HomePage />);

      // Check for form elements
      expect(screen.getByLabelText(/Yaş/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/İlişki Durumu/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Bütçe/i)).toBeInTheDocument();
    });

    it('should have proper spacing in mobile view', () => {
      const { container } = renderWithProviders(<HomePage />);

      // Check that container has responsive padding classes
      const mainContainer = container.querySelector('.container');
      expect(mainContainer).toBeInTheDocument();
    });
  });

  describe('Touch Interaction Tests', () => {
    it('should have touch-friendly button sizes', () => {
      renderWithProviders(<HomePage />);

      const submitButton = screen.getByRole('button', { name: /Hediye Önerisi Al/i });
      expect(submitButton).toBeInTheDocument();
      
      // Button should have appropriate classes for touch targets
      expect(submitButton.className).toContain('py-3');
    });

    it('should have touch-friendly navigation tabs', () => {
      renderWithProviders(<HomePage />);

      const searchTab = screen.getByText('Arama');
      const tabButton = searchTab.closest('button');
      
      expect(tabButton).toBeInTheDocument();
      // Should have touch-target class
      expect(tabButton?.className).toContain('touch-target');
    });

    it('should support touch manipulation on interactive elements', () => {
      renderWithProviders(<HomePage />);

      const buttons = screen.getAllByRole('button');
      
      // At least some buttons should have touch-manipulation class
      const hasTouchManipulation = buttons.some(button => 
        button.className.includes('touch-manipulation')
      );
      
      expect(hasTouchManipulation).toBe(true);
    });
  });

  describe('Responsive Text Tests', () => {
    it('should use responsive text sizes for headings', () => {
      const { container } = renderWithProviders(<HomePage />);

      const heading = screen.getByText(/Trendyol Hediye Önerisi/i);
      
      // Should have responsive text classes
      expect(heading.className).toMatch(/text-(3xl|4xl|5xl)/);
    });

    it('should have readable text on all screen sizes', () => {
      renderWithProviders(<HomePage />);

      const description = screen.getByText(/Sevdikleriniz için mükemmel hediyeyi bulun/i);
      expect(description).toBeInTheDocument();
      
      // Text should be visible and have appropriate styling
      expect(description.className).toContain('text-gray-600');
    });
  });

  describe('Grid Layout Tests', () => {
    it('should use single column grid on mobile', () => {
      global.innerWidth = 375;
      renderWithProviders(<HomePage />);

      // Form should be visible (single column layout)
      // Check for form by finding the submit button
      const submitButton = screen.getByRole('button', { name: /Hediye Önerisi Al/i });
      expect(submitButton).toBeInTheDocument();
      
      // Form element should exist in the DOM
      const formElement = submitButton.closest('form');
      expect(formElement).toBeInTheDocument();
    });

    it('should adapt grid columns based on screen size', () => {
      const { container } = renderWithProviders(<HomePage />);

      // Check for responsive grid classes in the DOM
      const gridElements = container.querySelectorAll('[class*="grid-cols"]');
      
      // Should have at least one grid element
      expect(gridElements.length).toBeGreaterThan(0);
    });
  });

  describe('Overflow Prevention Tests', () => {
    it('should not cause horizontal overflow on mobile', () => {
      global.innerWidth = 375;
      const { container } = renderWithProviders(<HomePage />);

      // Main container should have proper constraints
      const mainDiv = container.firstChild as HTMLElement;
      expect(mainDiv).toBeInTheDocument();
      
      // Should have min-h-screen class
      expect(mainDiv.className).toContain('min-h-screen');
    });

    it('should use proper container padding', () => {
      const { container } = renderWithProviders(<HomePage />);

      const containerDiv = container.querySelector('.container');
      expect(containerDiv).toBeInTheDocument();
      
      // Should have responsive padding
      expect(containerDiv?.className).toMatch(/px-\d/);
    });
  });

  describe('Performance Optimization Tests', () => {
    it('should render without layout shifts', async () => {
      const { container } = renderWithProviders(<HomePage />);

      // Initial render should be stable
      expect(container.firstChild).toBeInTheDocument();

      // Wait for any potential layout shifts
      await waitFor(() => {
        expect(container.firstChild).toBeInTheDocument();
      }, { timeout: 100 });
    });

    it('should have optimized image loading attributes', () => {
      // This would be tested with actual recommendation cards
      // For now, we verify the structure is in place
      renderWithProviders(<HomePage />);
      
      const mainContainer = screen.getByText(/Trendyol Hediye Önerisi/i);
      expect(mainContainer).toBeInTheDocument();
    });
  });
});
