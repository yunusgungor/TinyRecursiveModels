import { memo } from 'react';
import { ToolSelectionCard, ToolSelectionCardProps } from './ToolSelectionCard';
import { ThinkingStepsTimeline, ThinkingStepsTimelineProps } from './ThinkingStepsTimeline';
import { ConfidenceIndicator, ConfidenceIndicatorProps } from './ConfidenceIndicator';
import { GiftRecommendationCard, GiftRecommendationCardProps } from './GiftRecommendationCard';

/**
 * Memoized versions of expensive components
 * Prevents unnecessary re-renders when props haven't changed
 */

/**
 * Memoized ToolSelectionCard
 * Only re-renders when toolSelection prop changes
 */
export const MemoizedToolSelectionCard = memo<ToolSelectionCardProps>(
  ToolSelectionCard,
  (prevProps, nextProps) => {
    // Custom comparison: only re-render if toolSelection array changes
    return JSON.stringify(prevProps.toolSelection) === JSON.stringify(nextProps.toolSelection);
  }
);

MemoizedToolSelectionCard.displayName = 'MemoizedToolSelectionCard';

/**
 * Memoized ThinkingStepsTimeline
 * Only re-renders when steps array changes
 */
export const MemoizedThinkingStepsTimeline = memo<ThinkingStepsTimelineProps>(
  ThinkingStepsTimeline,
  (prevProps, nextProps) => {
    // Custom comparison: only re-render if steps array changes
    return (
      JSON.stringify(prevProps.steps) === JSON.stringify(nextProps.steps) &&
      prevProps.onStepClick === nextProps.onStepClick
    );
  }
);

MemoizedThinkingStepsTimeline.displayName = 'MemoizedThinkingStepsTimeline';

/**
 * Memoized ConfidenceIndicator
 * Only re-renders when confidence value changes
 */
export const MemoizedConfidenceIndicator = memo<ConfidenceIndicatorProps>(
  ConfidenceIndicator,
  (prevProps, nextProps) => {
    // Custom comparison: only re-render if confidence changes significantly
    return (
      Math.abs(prevProps.confidence - nextProps.confidence) < 0.01 &&
      prevProps.onClick === nextProps.onClick
    );
  }
);

MemoizedConfidenceIndicator.displayName = 'MemoizedConfidenceIndicator';

/**
 * Memoized GiftRecommendationCard
 * Only re-renders when recommendation data changes
 */
export const MemoizedGiftRecommendationCard = memo<GiftRecommendationCardProps>(
  GiftRecommendationCard,
  (prevProps, nextProps) => {
    // Custom comparison: check all relevant props
    return (
      prevProps.recommendation.gift.id === nextProps.recommendation.gift.id &&
      prevProps.recommendation.confidence === nextProps.recommendation.confidence &&
      prevProps.isSelected === nextProps.isSelected &&
      JSON.stringify(prevProps.recommendation.reasoning) === JSON.stringify(nextProps.recommendation.reasoning) &&
      JSON.stringify(prevProps.toolResults) === JSON.stringify(nextProps.toolResults)
    );
  }
);

MemoizedGiftRecommendationCard.displayName = 'MemoizedGiftRecommendationCard';
