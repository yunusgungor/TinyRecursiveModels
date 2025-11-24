import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { CategoryMatchingChart } from '../CategoryMatchingChart';
import { AttentionWeightsChart } from '../AttentionWeightsChart';
import { ReasoningPanel } from '../ReasoningPanel';
import type { CategoryMatchingReasoning, AttentionWeights, ReasoningTrace, GiftItem, UserProfile } from '@/types/reasoning';

/**
 * Integration Tests for Responsive Design
 * Tests mobile, tablet, and desktop viewport rendering
 * Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
 */

describe('Responsive Design Integration Tests', () => {
  let originalInnerWidth: number;
  let originalMatchMedia: typeof window.matchMedia;

  beforeEach(() => {
    originalInnerWidth = window.innerWidth;
    originalMatchMedia = window.matchMedia;
  });

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    });
    window.matchMedia = originalMatchMedia;
  });

  // Helper to mock window.matchMedia
  const mockMatchMedia = (width: number) => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: width,
    });

    window.matchMedia = (query: string) => ({
      matches: query === '(max-width: 767px)' ? width <= 767 : 
               query === '(min-width: 768px) and (max-width: 1023px)' ? (width >= 768 && width <= 1023) :
               query === '(min-width: 1024px)' ? width >= 1024 : false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => true,
    });
  };

  const mockCategories: CategoryMatchingReasoning[] = [
    {
      category_name: 'Electronics',
      score: 0.85,
      reasons: ['Hobby match', 'Age appropriate'],
      feature_contributions: { hobby: 0.9, age: 0.8 },
    },
    {
      category_name: 'Books',
      score: 0.65,
      reasons: ['Interest match'],
      feature_contributions: { hobby: 0.7 },
    },
    {
      category_name: 'Sports',
      score: 0.45,
      reasons: ['Occasional interest'],
      feature_contributions: { occasion: 0.5 },
    },
  ];

  const mockAttentionWeights: AttentionWeights = {
    user_features: {
      hobbies: 0.4,
      budget: 0.3,
      age: 0.2,
      occasion: 0.1,
    },
    gift_features: {
      category: 0.5,
      price: 0.3,
      rating: 0.2,
    },
  };

  const mockGift: GiftItem = {
    id: 'test-gift-1',
    name: 'Test Gift',
    price: 100,
    image_url: 'https://example.com/image.jpg',
    category: 'Electronics',
    rating: 4.5,
  };

  const mockUserProfile: UserProfile = {
    hobbies: ['reading', 'gaming'],
    budget: 150,
    age: 30,
    occasion: 'birthday',
  };

  const mockReasoningTrace: ReasoningTrace = {
    tool_selection: [{
      name: 'Review Analysis',
      selected: true,
      score: 0.85,
      reason: 'High rating match',
      confidence: 0.9,
      priority: 1,
    }],
    category_matching: mockCategories,
    attention_weights: mockAttentionWeights,
    thinking_steps: [{
      step: 1,
      action: 'Analyze user profile',
      result: 'Profile analyzed successfully',
      insight: 'User prefers tech gifts',
    }],
  };

  describe('Mobile viewport rendering (320px - 767px)', () => {
    it('should render CategoryMatchingChart in mobile layout', () => {
      mockMatchMedia(375); // iPhone size

      const { container } = render(
        <CategoryMatchingChart categories={mockCategories} />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Should have responsive classes
      const chartContainer = container.querySelector('.mb-4');
      expect(chartContainer).toBeTruthy();
    });

    it('should render AttentionWeightsChart with mobile-optimized dimensions', () => {
      mockMatchMedia(375);

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={() => {}}
        />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Chart type toggle should be present
      const toggleButtons = container.querySelectorAll('button[aria-pressed]');
      expect(toggleButtons.length).toBe(2);
    });

    it('should render ReasoningPanel as full-screen modal on mobile', () => {
      mockMatchMedia(375);

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection', 'category_matching', 'attention_weights', 'thinking_steps']}
          onFilterChange={() => {}}
        />
      );

      // Panel should be rendered in portal
      const panel = document.body.querySelector('.reasoning-panel');
      expect(panel).toBeTruthy();

      // Should have full height class for mobile
      if (panel) {
        expect(panel.className).toContain('h-full');
      }
    });
  });

  describe('Tablet viewport rendering (768px - 1023px)', () => {
    it('should render CategoryMatchingChart in tablet layout', () => {
      mockMatchMedia(768); // iPad size

      const { container } = render(
        <CategoryMatchingChart categories={mockCategories} />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Should have chart
      const chartContainer = container.querySelector('.mb-4');
      expect(chartContainer).toBeTruthy();
    });

    it('should render AttentionWeightsChart in tablet layout', () => {
      mockMatchMedia(768);

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={() => {}}
        />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Charts should be present
      const charts = container.querySelectorAll('.mb-6, div:last-child');
      expect(charts.length).toBeGreaterThan(0);
    });

    it('should render ReasoningPanel in tablet layout', () => {
      mockMatchMedia(768);

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['category_matching', 'attention_weights']}
          onFilterChange={() => {}}
        />
      );

      // Panel should be rendered
      const panel = document.body.querySelector('.reasoning-panel');
      expect(panel).toBeTruthy();
    });
  });

  describe('Desktop viewport rendering (1024px+)', () => {
    it('should render CategoryMatchingChart in desktop layout', () => {
      mockMatchMedia(1920); // Full HD

      const { container } = render(
        <CategoryMatchingChart categories={mockCategories} />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Should have chart
      const chartContainer = container.querySelector('.mb-4');
      expect(chartContainer).toBeTruthy();
    });

    it('should render AttentionWeightsChart in desktop layout', () => {
      mockMatchMedia(1920);

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={() => {}}
        />
      );

      // Component should be rendered
      const component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Chart type toggle should be present
      const toggleButtons = container.querySelectorAll('button[aria-pressed]');
      expect(toggleButtons.length).toBe(2);
    });

    it('should render ReasoningPanel in desktop layout', () => {
      mockMatchMedia(1920);

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection', 'category_matching', 'attention_weights', 'thinking_steps']}
          onFilterChange={() => {}}
        />
      );

      // Panel should be rendered
      const panel = document.body.querySelector('.reasoning-panel');
      expect(panel).toBeTruthy();

      // Should have max height class for desktop
      if (panel) {
        expect(panel.className).toMatch(/max-h-/);
      }
    });
  });

  describe('Responsive behavior across viewports', () => {
    it('should adapt CategoryMatchingChart when viewport changes', () => {
      // Start with mobile
      mockMatchMedia(375);
      const { container, rerender } = render(
        <CategoryMatchingChart categories={mockCategories} />
      );

      let component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();

      // Change to desktop
      mockMatchMedia(1920);
      rerender(<CategoryMatchingChart categories={mockCategories} />);

      component = container.querySelector('[role="region"]');
      expect(component).toBeTruthy();
    });

    it('should maintain functionality across all viewports', () => {
      const viewports = [375, 768, 1024, 1920];

      viewports.forEach((width) => {
        mockMatchMedia(width);

        const { container } = render(
          <AttentionWeightsChart
            attentionWeights={mockAttentionWeights}
            chartType="bar"
            onChartTypeChange={() => {}}
          />
        );

        // Component should always render
        const component = container.querySelector('[role="region"]');
        expect(component).toBeTruthy();

        // Toggle buttons should always be present
        const toggleButtons = container.querySelectorAll('button[aria-pressed]');
        expect(toggleButtons.length).toBe(2);
      });
    });
  });
});
