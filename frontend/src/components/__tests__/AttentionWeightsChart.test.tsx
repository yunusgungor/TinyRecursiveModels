import { describe, test, expect, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { AttentionWeightsChart } from '../AttentionWeightsChart';
import { AttentionWeights, ChartType } from '@/types/reasoning';

/**
 * Unit tests for AttentionWeightsChart component
 */

describe('AttentionWeightsChart', () => {
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

  describe('Bar chart rendering', () => {
    test('renders attention weights chart with title', () => {
      const onChartTypeChange = vi.fn();
      render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(screen.getByText('Attention Weights')).toBeInTheDocument();
    });

    test('renders user features section', () => {
      const onChartTypeChange = vi.fn();
      render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(screen.getByText('Kullanıcı Özellikleri')).toBeInTheDocument();
    });

    test('renders gift features section', () => {
      const onChartTypeChange = vi.fn();
      render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(screen.getByText('Hediye Özellikleri')).toBeInTheDocument();
    });

    test('displays bar chart when chartType is bar', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Check for BarChart elements (Recharts renders SVG)
      const svgElements = container.querySelectorAll('svg');
      expect(svgElements.length).toBeGreaterThan(0);
    });

    test('bar chart button is active when chartType is bar', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const barButton = container.querySelector('[aria-label="Bar grafik"]');
      expect(barButton).toHaveAttribute('aria-pressed', 'true');

      const radarButton = container.querySelector('[aria-label="Radar grafik"]');
      expect(radarButton).toHaveAttribute('aria-pressed', 'false');
    });
  });

  describe('Radar chart rendering', () => {
    test('displays radar chart when chartType is radar', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Check for RadarChart elements (Recharts renders SVG)
      const svgElements = container.querySelectorAll('svg');
      expect(svgElements.length).toBeGreaterThan(0);
    });

    test('radar chart button is active when chartType is radar', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const barButton = container.querySelector('[aria-label="Bar grafik"]');
      expect(barButton).toHaveAttribute('aria-pressed', 'false');

      const radarButton = container.querySelector('[aria-label="Radar grafik"]');
      expect(radarButton).toHaveAttribute('aria-pressed', 'true');
    });
  });

  describe('Chart type toggle', () => {
    test('calls onChartTypeChange when bar button is clicked', async () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const barButton = container.querySelector('[aria-label="Bar grafik"]') as HTMLElement;

      await act(async () => {
        barButton.click();
      });

      expect(onChartTypeChange).toHaveBeenCalledTimes(1);
      expect(onChartTypeChange).toHaveBeenCalledWith('bar');
    });

    test('calls onChartTypeChange when radar button is clicked', async () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const radarButton = container.querySelector('[aria-label="Radar grafik"]') as HTMLElement;

      await act(async () => {
        radarButton.click();
      });

      expect(onChartTypeChange).toHaveBeenCalledTimes(1);
      expect(onChartTypeChange).toHaveBeenCalledWith('radar');
    });

    test('updates chart type when prop changes', () => {
      const onChartTypeChange = vi.fn();
      const { container, rerender } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      let barButton = container.querySelector('[aria-label="Bar grafik"]');
      expect(barButton).toHaveAttribute('aria-pressed', 'true');

      rerender(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      barButton = container.querySelector('[aria-label="Bar grafik"]');
      const radarButton = container.querySelector('[aria-label="Radar grafik"]');
      expect(barButton).toHaveAttribute('aria-pressed', 'false');
      expect(radarButton).toHaveAttribute('aria-pressed', 'true');
    });
  });

  describe('Tooltip display', () => {
    test('renders charts with tooltip support', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Recharts tooltips are rendered as part of the chart
      // We can verify the component renders without errors
      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA role for the container', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const region = container.querySelector('[role="region"]');
      expect(region).toBeInTheDocument();
      expect(region).toHaveAttribute('aria-label', 'Attention weights bilgisi');
    });

    test('has proper ARIA role for chart type toggle group', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const toggleGroup = container.querySelector('[role="group"]');
      expect(toggleGroup).toBeInTheDocument();
      expect(toggleGroup).toHaveAttribute('aria-label', 'Grafik tipi seçimi');
    });

    test('chart type buttons have proper aria-pressed attributes', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const buttons = container.querySelectorAll('button[aria-pressed]');
      expect(buttons.length).toBe(2);

      buttons.forEach((button) => {
        const ariaPressed = button.getAttribute('aria-pressed');
        expect(['true', 'false']).toContain(ariaPressed);
      });
    });

    test('chart type buttons have proper aria-label attributes', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const barButton = container.querySelector('[aria-label="Bar grafik"]');
      const radarButton = container.querySelector('[aria-label="Radar grafik"]');

      expect(barButton).toBeInTheDocument();
      expect(radarButton).toBeInTheDocument();
    });

    test('charts have proper aria-label attributes', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const userChart = container.querySelector('[aria-label="Kullanıcı özellikleri attention weights grafiği"]');
      const giftChart = container.querySelector('[aria-label="Hediye özellikleri attention weights grafiği"]');

      expect(userChart).toBeInTheDocument();
      expect(giftChart).toBeInTheDocument();
    });
  });

  describe('Edge cases', () => {
    test('handles empty user features', () => {
      const onChartTypeChange = vi.fn();
      const emptyUserFeatures: AttentionWeights = {
        user_features: {},
        gift_features: {
          category: 0.5,
          price: 0.3,
          rating: 0.2,
        },
      };

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={emptyUserFeatures}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Should render without errors
      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });

    test('handles empty gift features', () => {
      const onChartTypeChange = vi.fn();
      const emptyGiftFeatures: AttentionWeights = {
        user_features: {
          hobbies: 0.4,
          budget: 0.3,
          age: 0.2,
          occasion: 0.1,
        },
        gift_features: {},
      };

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={emptyGiftFeatures}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Should render without errors
      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });

    test('handles single feature in user features', () => {
      const onChartTypeChange = vi.fn();
      const singleUserFeature: AttentionWeights = {
        user_features: {
          hobbies: 1.0,
        },
        gift_features: {
          category: 0.5,
          price: 0.3,
          rating: 0.2,
        },
      };

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={singleUserFeature}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });

    test('handles very small weights', () => {
      const onChartTypeChange = vi.fn();
      const smallWeights: AttentionWeights = {
        user_features: {
          hobbies: 0.01,
          budget: 0.01,
          age: 0.01,
          occasion: 0.97,
        },
        gift_features: {
          category: 0.01,
          price: 0.01,
          rating: 0.98,
        },
      };

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={smallWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });

    test('handles balanced weights', () => {
      const onChartTypeChange = vi.fn();
      const balancedWeights: AttentionWeights = {
        user_features: {
          hobbies: 0.25,
          budget: 0.25,
          age: 0.25,
          occasion: 0.25,
        },
        gift_features: {
          category: 0.33,
          price: 0.33,
          rating: 0.34,
        },
      };

      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={balancedWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      expect(container.querySelector('[role="region"]')).toBeTruthy();
    });
  });

  describe('Visual styling', () => {
    test('applies custom className when provided', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
          className="custom-class"
        />
      );

      const mainContainer = container.querySelector('[role="region"]');
      expect(mainContainer?.className).toContain('custom-class');
    });

    test('has proper styling classes', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const mainContainer = container.querySelector('[role="region"]');
      expect(mainContainer?.className).toContain('rounded-lg');
      expect(mainContainer?.className).toContain('border');
      expect(mainContainer?.className).toContain('bg-white');
      expect(mainContainer?.className).toContain('shadow-sm');
    });

    test('bar button has active styling when selected', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const barButton = container.querySelector('[aria-label="Bar grafik"]');
      expect(barButton?.className).toContain('bg-blue-600');
      expect(barButton?.className).toContain('text-white');
    });

    test('radar button has active styling when selected', () => {
      const onChartTypeChange = vi.fn();
      const { container } = render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="radar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      const radarButton = container.querySelector('[aria-label="Radar grafik"]');
      expect(radarButton?.className).toContain('bg-blue-600');
      expect(radarButton?.className).toContain('text-white');
    });
  });

  describe('Feature name formatting', () => {
    test('displays Turkish feature names', () => {
      const onChartTypeChange = vi.fn();
      render(
        <AttentionWeightsChart
          attentionWeights={mockAttentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Turkish translations should be used
      expect(screen.getByText('Kullanıcı Özellikleri')).toBeInTheDocument();
      expect(screen.getByText('Hediye Özellikleri')).toBeInTheDocument();
    });
  });
});
