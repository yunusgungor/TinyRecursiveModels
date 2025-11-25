import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import * as fc from 'fast-check';
import { GiftRecommendationCard } from '../GiftRecommendationCard';
import { ConfidenceIndicator } from '../ConfidenceIndicator';
import { ThinkingStepsTimeline } from '../ThinkingStepsTimeline';
import { CategoryMatchingChart } from '../CategoryMatchingChart';
import { AttentionWeightsChart } from '../AttentionWeightsChart';
import { ToolSelectionCard } from '../ToolSelectionCard';
import { EnhancedGiftRecommendation, ThinkingStep, CategoryMatchingReasoning, AttentionWeights, ToolSelectionReasoning } from '@/types/reasoning';

/**
 * Feature: frontend-reasoning-visualization, Property 58: ARIA labels presence
 * Validates: Requirements 15.1
 * 
 * For any reasoning component, the frontend should include ARIA labels and roles.
 */
describe('Property 58: ARIA labels presence', () => {
  it('should have ARIA labels on GiftRecommendationCard', () => {
    fc.assert(
      fc.property(
        fc.record({
          gift: fc.record({
            id: fc.string({ minLength: 1, maxLength: 20 }),
            name: fc.string({ minLength: 5, maxLength: 50 }),
            category: fc.string({ minLength: 3, maxLength: 20 }),
            price: fc.float({ min: 10, max: 10000 }),
            image_url: fc.webUrl(),
          }),
          reasoning: fc.array(fc.string({ minLength: 10, maxLength: 100 }), { minLength: 1, maxLength: 3 }),
          confidence: fc.float({ min: 0, max: 1 }),
        }),
        (recommendation) => {
          const { container } = render(
            <GiftRecommendationCard
              recommendation={recommendation as EnhancedGiftRecommendation}
            />
          );
          
          const article = container.querySelector('[role="article"]');
          expect(article).toBeTruthy();
          expect(article?.getAttribute('aria-label')).toContain(recommendation.gift.name);
          
          if (recommendation.reasoning.length > 0) {
            const reasoningRegion = container.querySelector('[role="region"][aria-label*="Reasoning"]');
            expect(reasoningRegion).toBeTruthy();
          }
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should have ARIA labels on ConfidenceIndicator', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1 }),
        (confidence) => {
          const { container } = render(
            <ConfidenceIndicator confidence={confidence} />
          );
          
          const confidenceElement = container.querySelector('[role="status"], [role="button"]');
          expect(confidenceElement).toBeTruthy();
          expect(confidenceElement?.getAttribute('aria-label')).toBeTruthy();
          expect(confidenceElement?.getAttribute('aria-label')).toContain('Güven skoru');
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should have ARIA labels on ThinkingStepsTimeline', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            step: fc.integer({ min: 1, max: 20 }),
            action: fc.string({ minLength: 10, maxLength: 50 }),
            result: fc.string({ minLength: 10, maxLength: 100 }),
            insight: fc.string({ minLength: 10, maxLength: 100 }),
          }),
          { minLength: 1, maxLength: 5 }
        ),
        (thinkingSteps) => {
          const { container } = render(
            <ThinkingStepsTimeline steps={thinkingSteps as ThinkingStep[]} />
          );
          
          const timelineRegion = container.querySelector('[role="region"]');
          expect(timelineRegion).toBeTruthy();
          expect(timelineRegion?.getAttribute('aria-label')).toContain('zaman çizelgesi');
          
          const timelineList = container.querySelector('[role="list"]');
          expect(timelineList).toBeTruthy();
          
          const timelineButtons = container.querySelectorAll('[role="button"]');
          expect(timelineButtons.length).toBe(thinkingSteps.length);
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should have ARIA labels on CategoryMatchingChart', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            category_name: fc.string({ minLength: 3, maxLength: 20 }),
            score: fc.float({ min: 0, max: 1 }),
            reasons: fc.array(fc.string({ minLength: 10, maxLength: 50 }), { minLength: 1, maxLength: 2 }),
            feature_contributions: fc.dictionary(
              fc.constantFrom('hobby', 'age', 'budget'),
              fc.float({ min: 0, max: 1 })
            ),
          }),
          { minLength: 3, maxLength: 5 }
        ),
        (categories) => {
          const { container } = render(
            <CategoryMatchingChart categories={categories as CategoryMatchingReasoning[]} />
          );
          
          const categoryRegion = container.querySelector('[role="region"]');
          expect(categoryRegion).toBeTruthy();
          expect(categoryRegion?.getAttribute('aria-label')).toContain('Kategori');
          
          const categoryChart = container.querySelector('[role="img"]');
          expect(categoryChart).toBeTruthy();
          const ariaLabel = categoryChart?.getAttribute('aria-label');
          expect(ariaLabel).toBeTruthy();
          // Check for Turkish "grafik" or "grafiği" (with Turkish characters)
          expect(ariaLabel).toMatch(/grafi[kğ]/i);
          
          const categoryList = container.querySelector('[role="list"]');
          expect(categoryList).toBeTruthy();
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should have ARIA labels on AttentionWeightsChart', () => {
    fc.assert(
      fc.property(
        fc.record({
          user_features: fc.record({
            hobbies: fc.float({ min: 0, max: 1 }),
            budget: fc.float({ min: 0, max: 1 }),
            age: fc.float({ min: 0, max: 1 }),
            occasion: fc.float({ min: 0, max: 1 }),
          }),
          gift_features: fc.record({
            category: fc.float({ min: 0, max: 1 }),
            price: fc.float({ min: 0, max: 1 }),
            rating: fc.float({ min: 0, max: 1 }),
          }),
        }),
        (attentionWeights) => {
          const { container } = render(
            <AttentionWeightsChart
              attentionWeights={attentionWeights as AttentionWeights}
              chartType="bar"
              onChartTypeChange={() => {}}
            />
          );
          
          const attentionRegion = container.querySelector('[role="region"]');
          expect(attentionRegion).toBeTruthy();
          expect(attentionRegion?.getAttribute('aria-label')).toContain('Attention weights');
          
          const toggleGroup = container.querySelector('[role="group"]');
          expect(toggleGroup).toBeTruthy();
          expect(toggleGroup?.getAttribute('aria-label')).toContain('Grafik tipi');
          
          const attentionCharts = container.querySelectorAll('[role="img"]');
          expect(attentionCharts.length).toBeGreaterThanOrEqual(2);
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should have ARIA labels on ToolSelectionCard', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            name: fc.constantFrom('review_analysis', 'trend_analysis', 'inventory_check'),
            selected: fc.boolean(),
            score: fc.float({ min: 0, max: 1 }),
            reason: fc.string({ minLength: 10, maxLength: 100 }),
            confidence: fc.float({ min: 0, max: 1 }),
            priority: fc.integer({ min: 1, max: 10 }),
          }),
          { minLength: 1, maxLength: 3 }
        ),
        (toolSelection) => {
          const { container } = render(
            <ToolSelectionCard toolSelection={toolSelection as ToolSelectionReasoning[]} />
          );
          
          const toolRegion = container.querySelector('[role="region"]');
          expect(toolRegion).toBeTruthy();
          expect(toolRegion?.getAttribute('aria-label')).toContain('Tool');
          
          const toolList = container.querySelector('[role="list"]');
          expect(toolList).toBeTruthy();
          
          const toolItems = container.querySelectorAll('[role="listitem"]');
          expect(toolItems.length).toBe(toolSelection.length);
          
          toolItems.forEach((item) => {
            expect(item.getAttribute('aria-label')).toBeTruthy();
          });
        }
      ),
      { numRuns: 50 }
    );
  });
});

/**
 * Feature: frontend-reasoning-visualization, Property 60: Keyboard navigation
 * Validates: Requirements 15.3
 * 
 * For any keyboard navigation usage, the frontend should provide access to all interactive elements.
 */
describe('Property 60: Keyboard navigation', () => {
  it('should make GiftRecommendationCard keyboard accessible', () => {
    fc.assert(
      fc.property(
        fc.record({
          gift: fc.record({
            id: fc.string({ minLength: 1, maxLength: 20 }),
            name: fc.string({ minLength: 5, maxLength: 50 }),
            category: fc.string({ minLength: 3, maxLength: 20 }),
            price: fc.float({ min: 10, max: 10000 }),
            image_url: fc.webUrl(),
          }),
          reasoning: fc.array(fc.string({ minLength: 10, maxLength: 100 }), { minLength: 3, maxLength: 5 }),
          confidence: fc.float({ min: 0, max: 1 }),
        }),
        (recommendation) => {
          const { container } = render(
            <GiftRecommendationCard
              recommendation={recommendation as EnhancedGiftRecommendation}
              onShowDetails={() => {}}
              onSelect={() => {}}
            />
          );
          
          const showDetailsButton = container.querySelector('button[aria-label*="Detaylı analiz"]');
          expect(showDetailsButton).toBeTruthy();
          expect(showDetailsButton?.getAttribute('tabIndex')).not.toBe('-1');
          
          const expandButton = container.querySelector('button[aria-expanded]');
          if (expandButton) {
            expect(expandButton.getAttribute('tabIndex')).not.toBe('-1');
          }
          
          const checkbox = container.querySelector('input[type="checkbox"]');
          if (checkbox) {
            expect(checkbox.getAttribute('tabIndex')).not.toBe('-1');
          }
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should make ConfidenceIndicator keyboard accessible', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1 }),
        (confidence) => {
          const { container } = render(
            <ConfidenceIndicator confidence={confidence} onClick={() => {}} />
          );
          
          const confidenceElement = container.querySelector('[role="button"]');
          expect(confidenceElement).toBeTruthy();
          expect(confidenceElement?.getAttribute('tabIndex')).toBe('0');
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should make ThinkingStepsTimeline keyboard accessible', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            step: fc.integer({ min: 1, max: 20 }),
            action: fc.string({ minLength: 10, maxLength: 50 }),
            result: fc.string({ minLength: 10, maxLength: 100 }),
            insight: fc.string({ minLength: 10, maxLength: 100 }),
          }),
          { minLength: 1, maxLength: 5 }
        ),
        (thinkingSteps) => {
          const { container } = render(
            <ThinkingStepsTimeline steps={thinkingSteps as ThinkingStep[]} />
          );
          
          const timelineButtons = container.querySelectorAll('[role="button"]');
          timelineButtons.forEach((button) => {
            expect(button.getAttribute('tabIndex')).toBe('0');
            expect(button.getAttribute('aria-expanded')).toBeTruthy();
          });
        }
      ),
      { numRuns: 50 }
    );
  });
});

/**
 * Feature: frontend-reasoning-visualization, Property 62: Color-blind friendly design
 * Validates: Requirements 15.5
 * 
 * For any color usage, the frontend should also provide visual cues beyond color (patterns, icons).
 */
describe('Property 62: Color-blind friendly design', () => {
  it('should provide text labels in ConfidenceIndicator', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
        (confidence) => {
          const { container } = render(
            <ConfidenceIndicator confidence={confidence} />
          );
          
          const confidenceText = container.textContent;
          expect(confidenceText).toMatch(/Yüksek Güven|Orta Güven|Düşük Güven/);
          expect(confidenceText).toMatch(/\d+%/);
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should provide score percentages in CategoryMatchingChart', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            category_name: fc.string({ minLength: 3, maxLength: 20 }),
            score: fc.float({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
            reasons: fc.array(fc.string({ minLength: 10, maxLength: 50 }), { minLength: 1, maxLength: 2 }),
            feature_contributions: fc.dictionary(
              fc.constantFrom('hobby', 'age', 'budget'),
              fc.float({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true })
            ),
          }),
          { minLength: 3, maxLength: 5 }
        ),
        (categories) => {
          const { container } = render(
            <CategoryMatchingChart categories={categories as CategoryMatchingReasoning[]} />
          );
          
          const categoryItems = container.querySelectorAll('[role="listitem"]');
          categoryItems.forEach((item) => {
            const itemText = item.textContent;
            expect(itemText).toMatch(/\d+%/);
          });
        }
      ),
      { numRuns: 50 }
    );
  });

  it('should provide icons in ToolSelectionCard', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            name: fc.constantFrom('review_analysis', 'trend_analysis', 'inventory_check'),
            selected: fc.boolean(),
            score: fc.float({ min: 0, max: 1 }),
            reason: fc.string({ minLength: 10, maxLength: 100 }),
            confidence: fc.float({ min: 0, max: 1 }),
            priority: fc.integer({ min: 1, max: 10 }),
          }),
          { minLength: 2, maxLength: 3 }
        ),
        (toolSelection) => {
          const { container } = render(
            <ToolSelectionCard toolSelection={toolSelection as ToolSelectionReasoning[]} />
          );
          
          const toolItems = container.querySelectorAll('[role="listitem"]');
          toolItems.forEach((item) => {
            const icon = item.querySelector('svg');
            expect(icon).toBeTruthy();
          });
          
          const selectedTools = toolSelection.filter(t => t.selected);
          if (selectedTools.length > 0) {
            const checkmarks = container.querySelectorAll('path[d*="M5 13l4 4L19 7"]');
            expect(checkmarks.length).toBeGreaterThan(0);
          }
          
          const unselectedTools = toolSelection.filter(t => !t.selected);
          if (unselectedTools.length > 0) {
            const xIcons = container.querySelectorAll('path[d*="M6 18L18 6M6 6l12 12"]');
            expect(xIcons.length).toBeGreaterThan(0);
          }
        }
      ),
      { numRuns: 50 }
    );
  });
});
