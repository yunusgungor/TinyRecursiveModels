/**
 * Storybook stories for loading state components
 */

import type { Meta, StoryObj } from '@storybook/react';
import {
  Spinner,
  GiftCardSkeleton,
  ToolSelectionSkeleton,
  CategoryChartSkeleton,
  AttentionWeightsSkeleton,
  ThinkingStepsSkeleton,
  ReasoningPanelSkeleton,
  LoadingOverlay,
} from './LoadingStates';

const meta: Meta = {
  title: 'Components/Loading States',
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
};

export default meta;

export const SpinnerSmall: StoryObj<typeof Spinner> = {
  render: () => <Spinner size="sm" />,
};

export const SpinnerMedium: StoryObj<typeof Spinner> = {
  render: () => <Spinner size="md" />,
};

export const SpinnerLarge: StoryObj<typeof Spinner> = {
  render: () => <Spinner size="lg" />,
};

export const GiftCardSkeletonStory: StoryObj<typeof GiftCardSkeleton> = {
  render: () => (
    <div className="w-80">
      <GiftCardSkeleton />
    </div>
  ),
};

export const ToolSelectionSkeletonStory: StoryObj<typeof ToolSelectionSkeleton> = {
  render: () => (
    <div className="w-96">
      <ToolSelectionSkeleton />
    </div>
  ),
};

export const CategoryChartSkeletonStory: StoryObj<typeof CategoryChartSkeleton> = {
  render: () => (
    <div className="w-96">
      <CategoryChartSkeleton />
    </div>
  ),
};

export const AttentionWeightsSkeletonStory: StoryObj<typeof AttentionWeightsSkeleton> = {
  render: () => (
    <div className="w-96">
      <AttentionWeightsSkeleton />
    </div>
  ),
};

export const ThinkingStepsSkeletonStory: StoryObj<typeof ThinkingStepsSkeleton> = {
  render: () => (
    <div className="w-96">
      <ThinkingStepsSkeleton />
    </div>
  ),
};

export const ReasoningPanelSkeletonStory: StoryObj<typeof ReasoningPanelSkeleton> = {
  render: () => (
    <div className="w-full max-w-4xl">
      <ReasoningPanelSkeleton />
    </div>
  ),
};

export const LoadingOverlayStory: StoryObj<typeof LoadingOverlay> = {
  render: () => <LoadingOverlay message="Öneriler yükleniyor..." />,
  parameters: {
    layout: 'fullscreen',
  },
};
