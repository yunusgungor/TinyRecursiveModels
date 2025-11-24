import type { Meta, StoryObj } from '@storybook/react';
import { ConfidenceIndicator } from './ConfidenceIndicator';

/**
 * The ConfidenceIndicator component displays a confidence score with color-coded styling
 * and an optional click handler for showing detailed explanations.
 * 
 * ## Color Coding
 * - **Green** (>0.8): High confidence
 * - **Yellow** (0.5-0.8): Medium confidence
 * - **Red** (<0.5): Low confidence
 * 
 * ## Accessibility
 * - ARIA labels for screen readers
 * - Keyboard navigable (Tab, Enter, Space)
 * - Color-blind friendly with text labels
 */
const meta = {
  title: 'Components/ConfidenceIndicator',
  component: ConfidenceIndicator,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    confidence: {
      control: { type: 'range', min: 0, max: 1, step: 0.01 },
      description: 'Confidence score between 0 and 1',
    },
    onClick: {
      action: 'clicked',
      description: 'Optional click handler for showing explanation modal',
    },
  },
} satisfies Meta<typeof ConfidenceIndicator>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * High confidence indicator (>0.8) with green styling
 */
export const HighConfidence: Story = {
  args: {
    confidence: 0.92,
  },
};

/**
 * Medium confidence indicator (0.5-0.8) with yellow styling
 */
export const MediumConfidence: Story = {
  args: {
    confidence: 0.65,
  },
};

/**
 * Low confidence indicator (<0.5) with red styling
 */
export const LowConfidence: Story = {
  args: {
    confidence: 0.35,
  },
};

/**
 * Clickable indicator with onClick handler
 */
export const Clickable: Story = {
  args: {
    confidence: 0.85,
    onClick: () => alert('Confidence explanation modal would open here'),
  },
};

/**
 * Boundary value at 0.8 (should be medium)
 */
export const BoundaryHigh: Story = {
  args: {
    confidence: 0.8,
  },
};

/**
 * Boundary value at 0.5 (should be medium)
 */
export const BoundaryLow: Story = {
  args: {
    confidence: 0.5,
  },
};

/**
 * Very high confidence (100%)
 */
export const Perfect: Story = {
  args: {
    confidence: 1.0,
  },
};

/**
 * Very low confidence (near 0%)
 */
export const VeryLow: Story = {
  args: {
    confidence: 0.05,
  },
};

/**
 * Multiple indicators showing different confidence levels
 */
export const MultipleIndicators: Story = {
  args: {
    confidence: 0.75,
  },
  render: () => (
    <div className="flex flex-col gap-4">
      <ConfidenceIndicator confidence={0.95} />
      <ConfidenceIndicator confidence={0.85} />
      <ConfidenceIndicator confidence={0.75} />
      <ConfidenceIndicator confidence={0.65} />
      <ConfidenceIndicator confidence={0.55} />
      <ConfidenceIndicator confidence={0.45} />
      <ConfidenceIndicator confidence={0.35} />
      <ConfidenceIndicator confidence={0.25} />
      <ConfidenceIndicator confidence={0.15} />
    </div>
  ),
};

/**
 * Indicators with custom className
 */
export const WithCustomClass: Story = {
  args: {
    confidence: 0.75,
    className: 'shadow-lg',
  },
};
