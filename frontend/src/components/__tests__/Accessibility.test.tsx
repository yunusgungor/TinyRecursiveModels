import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GiftRecommendationCard } from '../GiftRecommendationCard';
import { ConfidenceIndicator } from '../ConfidenceIndicator';
import { ThinkingStepsTimeline } from '../ThinkingStepsTimeline';
import { CategoryMatchingChart } from '../CategoryMatchingChart';
import { AttentionWeightsChart } from '../AttentionWeightsChart';
import { ToolSelectionCard } from '../ToolSelectionCard';
import { ReasoningPanel } from '../ReasoningPanel';
import { EnhancedGiftRecommendation, ThinkingStep, CategoryMatchingReasoning, AttentionWeights, ToolSelectionReasoning, ReasoningTrace } from '@/types/reasoning';

describe('Accessibility Tests', () => {
  describe('Keyboard Navigation', () => {
    it('should navigate through GiftRecommendationCard with keyboard', async () => {
      const user = userEvent.setup();
      const onShowDetails = vi.fn();
      const onSelect = vi.fn();

      const recommendation: EnhancedGiftRecommendation = {
        gift: {
          id: '1',
          name: 'Test Gift',
          category: 'Electronics',
          price: 100,
          image_url: 'https://example.com/image.jpg',
        },
        reasoning: ['Great for tech enthusiasts', 'Within budget', 'Highly rated'],
        confidence: 0.85,
      };

      render(
        <GiftRecommendationCard
          recommendation={recommendation}
          onShowDetails={onShowDetails}
          onSelect={onSelect}
        />
      );

      // Tab to the "Show Details" button
      await user.tab();
      const showDetailsButton = screen.getByRole('button', { name: /Detaylı analiz göster/i });
      expect(showDetailsButton).toHaveFocus();

      // Press Enter to trigger the button
      await user.keyboard('{Enter}');
      expect(onShowDetails).toHaveBeenCalledTimes(1);

      // Tab to the checkbox
      await user.tab();
      const checkbox = screen.getByRole('checkbox', { name: /Karşılaştırma için seç/i });
      expect(checkbox).toHaveFocus();

      // Press Space to toggle the checkbox
      await user.keyboard(' ');
      expect(onSelect).toHaveBeenCalledTimes(1);
    });

    it('should navigate through ThinkingStepsTimeline with keyboard', async () => {
      const user = userEvent.setup();
      const onStepClick = vi.fn();

      const steps: ThinkingStep[] = [
        {
          step: 1,
          action: 'Analyze user profile',
          result: 'Identified hobbies',
          insight: 'User prefers outdoor activities',
        },
        {
          step: 2,
          action: 'Filter categories',
          result: 'Selected 5 categories',
          insight: 'Sports and outdoor categories match',
        },
      ];

      render(<ThinkingStepsTimeline steps={steps} onStepClick={onStepClick} />);

      // Tab to the first step
      await user.tab();
      const firstStep = screen.getByRole('button', { name: /Adım 1: Analyze user profile/i });
      expect(firstStep).toHaveFocus();

      // Press Enter to expand the step
      await user.keyboard('{Enter}');
      expect(onStepClick).toHaveBeenCalledWith(steps[0]);

      // Tab to the second step
      await user.tab();
      const secondStep = screen.getByRole('button', { name: /Adım 2: Filter categories/i });
      expect(secondStep).toHaveFocus();

      // Press Space to expand the step
      await user.keyboard(' ');
      expect(onStepClick).toHaveBeenCalledWith(steps[1]);
    });

    it('should navigate through CategoryMatchingChart with keyboard', async () => {
      const user = userEvent.setup();
      const onCategoryClick = vi.fn();

      const categories: CategoryMatchingReasoning[] = [
        {
          category_name: 'Electronics',
          score: 0.85,
          reasons: ['Hobby match', 'Age appropriate'],
          feature_contributions: { hobby: 0.9, age: 0.8 },
        },
        {
          category_name: 'Books',
          score: 0.65,
          reasons: ['Budget friendly'],
          feature_contributions: { budget: 0.7 },
        },
        {
          category_name: 'Sports',
          score: 0.45,
          reasons: ['Partial match'],
          feature_contributions: { hobby: 0.5 },
        },
      ];

      render(<CategoryMatchingChart categories={categories} onCategoryClick={onCategoryClick} />);

      // Tab to the first category
      await user.tab();
      const firstCategory = screen.getByRole('listitem', { name: /Electronics: 85% skor/i });
      expect(firstCategory).toHaveFocus();

      // Press Enter to expand the category
      await user.keyboard('{Enter}');
      expect(onCategoryClick).toHaveBeenCalledWith(categories[0]);
    });

    it('should navigate through AttentionWeightsChart toggle with keyboard', async () => {
      const user = userEvent.setup();
      const onChartTypeChange = vi.fn();

      const attentionWeights: AttentionWeights = {
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

      render(
        <AttentionWeightsChart
          attentionWeights={attentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Tab to the bar chart button
      await user.tab();
      const barButton = screen.getByRole('button', { name: /Bar grafik/i });
      expect(barButton).toHaveFocus();

      // Tab to the radar chart button
      await user.tab();
      const radarButton = screen.getByRole('button', { name: /Radar grafik/i });
      expect(radarButton).toHaveFocus();

      // Press Enter to switch to radar chart
      await user.keyboard('{Enter}');
      expect(onChartTypeChange).toHaveBeenCalledWith('radar');
    });
  });

  describe('Screen Reader Compatibility', () => {
    it('should have proper ARIA labels for GiftRecommendationCard', () => {
      const recommendation: EnhancedGiftRecommendation = {
        gift: {
          id: '1',
          name: 'Test Gift',
          category: 'Electronics',
          price: 100,
          image_url: 'https://example.com/image.jpg',
        },
        reasoning: ['Great for tech enthusiasts'],
        confidence: 0.85,
      };

      render(<GiftRecommendationCard recommendation={recommendation} />);

      // Check for article role and aria-label
      const article = screen.getByRole('article', { name: /Gift recommendation: Test Gift/i });
      expect(article).toBeInTheDocument();

      // Check for reasoning region
      const reasoningRegion = screen.getByRole('region', { name: /Reasoning information/i });
      expect(reasoningRegion).toBeInTheDocument();
    });

    it('should have proper ARIA labels for ConfidenceIndicator', () => {
      render(<ConfidenceIndicator confidence={0.85} />);

      // Check for status role and aria-label
      const confidenceStatus = screen.getByRole('status', { name: /Güven skoru: 85%, Yüksek Güven/i });
      expect(confidenceStatus).toBeInTheDocument();
      expect(confidenceStatus).toHaveAttribute('aria-live', 'polite');
    });

    it('should have proper ARIA labels for ThinkingStepsTimeline', () => {
      const steps: ThinkingStep[] = [
        {
          step: 1,
          action: 'Analyze user profile',
          result: 'Identified hobbies',
          insight: 'User prefers outdoor activities',
        },
      ];

      render(<ThinkingStepsTimeline steps={steps} />);

      // Check for region role and aria-label
      const timelineRegion = screen.getByRole('region', { name: /Düşünme adımları zaman çizelgesi/i });
      expect(timelineRegion).toBeInTheDocument();

      // Check for list role
      const timelineList = screen.getByRole('list');
      expect(timelineList).toBeInTheDocument();

      // Check for button with aria-expanded
      const stepButton = screen.getByRole('button', { name: /Adım 1: Analyze user profile/i });
      expect(stepButton).toHaveAttribute('aria-expanded');
    });

    it('should have proper ARIA labels for CategoryMatchingChart', () => {
      const categories: CategoryMatchingReasoning[] = [
        {
          category_name: 'Electronics',
          score: 0.85,
          reasons: ['Hobby match'],
          feature_contributions: { hobby: 0.9 },
        },
        {
          category_name: 'Books',
          score: 0.65,
          reasons: ['Budget friendly'],
          feature_contributions: { budget: 0.7 },
        },
        {
          category_name: 'Sports',
          score: 0.45,
          reasons: ['Partial match'],
          feature_contributions: { hobby: 0.5 },
        },
      ];

      render(<CategoryMatchingChart categories={categories} />);

      // Check for region role and aria-label
      const categoryRegion = screen.getByRole('region', { name: /Kategori eşleştirme bilgisi/i });
      expect(categoryRegion).toBeInTheDocument();

      // Check for chart with role="img" and aria-label
      const categoryChart = screen.getByRole('img', { name: /Kategori skorları bar grafiği/i });
      expect(categoryChart).toBeInTheDocument();

      // Check for list role
      const categoryList = screen.getByRole('list');
      expect(categoryList).toBeInTheDocument();
    });

    it('should have proper ARIA labels for AttentionWeightsChart', () => {
      const attentionWeights: AttentionWeights = {
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

      render(
        <AttentionWeightsChart
          attentionWeights={attentionWeights}
          chartType="bar"
          onChartTypeChange={() => {}}
        />
      );

      // Check for region role and aria-label
      const attentionRegion = screen.getByRole('region', { name: /Attention weights bilgisi/i });
      expect(attentionRegion).toBeInTheDocument();

      // Check for chart type toggle group
      const toggleGroup = screen.getByRole('group', { name: /Grafik tipi seçimi/i });
      expect(toggleGroup).toBeInTheDocument();

      // Check for charts with role="img"
      const charts = screen.getAllByRole('img');
      expect(charts.length).toBeGreaterThanOrEqual(2); // User and gift features
    });

    it('should have proper ARIA labels for ToolSelectionCard', () => {
      const toolSelection: ToolSelectionReasoning[] = [
        {
          name: 'review_analysis',
          selected: true,
          score: 0.85,
          reason: 'High rating match',
          confidence: 0.9,
          priority: 1,
        },
        {
          name: 'trend_analysis',
          selected: false,
          score: 0.45,
          reason: 'Low trend relevance',
          confidence: 0.4,
          priority: 2,
        },
      ];

      render(<ToolSelectionCard toolSelection={toolSelection} />);

      // Check for region role and aria-label
      const toolRegion = screen.getByRole('region', { name: /Tool seçim bilgisi/i });
      expect(toolRegion).toBeInTheDocument();

      // Check for list role
      const toolList = screen.getByRole('list');
      expect(toolList).toBeInTheDocument();

      // Check for list items with aria-labels
      const toolItems = screen.getAllByRole('listitem');
      expect(toolItems.length).toBe(2);
      toolItems.forEach((item) => {
        expect(item).toHaveAttribute('aria-label');
      });
    });
  });

  describe('Focus Management', () => {
    it('should manage focus when expanding ThinkingStepsTimeline', async () => {
      const user = userEvent.setup();

      const steps: ThinkingStep[] = [
        {
          step: 1,
          action: 'Analyze user profile',
          result: 'Identified hobbies',
          insight: 'User prefers outdoor activities',
        },
        {
          step: 2,
          action: 'Filter categories',
          result: 'Selected 5 categories',
          insight: 'Sports and outdoor categories match',
        },
      ];

      render(<ThinkingStepsTimeline steps={steps} />);

      // Click on the first step to expand it
      const firstStep = screen.getByRole('button', { name: /Adım 1: Analyze user profile/i });
      await user.click(firstStep);

      // Check that the step is expanded
      expect(firstStep).toHaveAttribute('aria-expanded', 'true');

      // Check that the insight is now visible
      expect(screen.getByText(/User prefers outdoor activities/i)).toBeInTheDocument();
    });

    it('should manage focus when clicking ConfidenceIndicator', async () => {
      const user = userEvent.setup();
      const onClick = vi.fn();

      render(<ConfidenceIndicator confidence={0.85} onClick={onClick} />);

      // Check that the indicator is focusable
      const indicator = screen.getByRole('button', { name: /Güven skoru: 85%, Yüksek Güven/i });
      expect(indicator).toHaveAttribute('tabIndex', '0');

      // Tab to the indicator
      await user.tab();
      expect(indicator).toHaveFocus();

      // Press Enter to trigger onClick
      await user.keyboard('{Enter}');
      expect(onClick).toHaveBeenCalledTimes(1);

      // Press Space to trigger onClick again
      await user.keyboard(' ');
      expect(onClick).toHaveBeenCalledTimes(2);
    });

    it('should maintain focus when toggling chart type in AttentionWeightsChart', async () => {
      const user = userEvent.setup();
      const onChartTypeChange = vi.fn();

      const attentionWeights: AttentionWeights = {
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

      render(
        <AttentionWeightsChart
          attentionWeights={attentionWeights}
          chartType="bar"
          onChartTypeChange={onChartTypeChange}
        />
      );

      // Click on the radar button
      const radarButton = screen.getByRole('button', { name: /Radar grafik/i });
      await user.click(radarButton);

      // Check that the button is pressed
      expect(radarButton).toHaveAttribute('aria-pressed', 'false'); // Still false because we didn't actually change the chart type
      expect(onChartTypeChange).toHaveBeenCalledWith('radar');
    });
  });
});
