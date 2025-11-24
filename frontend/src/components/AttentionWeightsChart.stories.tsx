import type { Meta, StoryObj } from '@storybook/react';
import { AttentionWeightsChart } from './AttentionWeightsChart';
import { useState } from 'react';
import { ChartType } from '@/types/reasoning';

const meta = {
  title: 'Components/AttentionWeightsChart',
  component: AttentionWeightsChart,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof AttentionWeightsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

// Wrapper component to handle state
const AttentionWeightsChartWithState = (args: any) => {
  const [chartType, setChartType] = useState<ChartType>(args.chartType || 'bar');
  
  return (
    <AttentionWeightsChart
      {...args}
      chartType={chartType}
      onChartTypeChange={setChartType}
    />
  );
};

export const Default: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
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
    },
    chartType: 'bar',
  },
};

export const BarChart: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.35,
        budget: 0.25,
        age: 0.25,
        occasion: 0.15,
      },
      gift_features: {
        category: 0.45,
        price: 0.35,
        rating: 0.2,
      },
    },
    chartType: 'bar',
  },
};

export const RadarChart: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.35,
        budget: 0.25,
        age: 0.25,
        occasion: 0.15,
      },
      gift_features: {
        category: 0.45,
        price: 0.35,
        rating: 0.2,
      },
    },
    chartType: 'radar',
  },
};

export const HighHobbyWeight: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.7,
        budget: 0.15,
        age: 0.1,
        occasion: 0.05,
      },
      gift_features: {
        category: 0.6,
        price: 0.25,
        rating: 0.15,
      },
    },
    chartType: 'bar',
  },
};

export const BalancedWeights: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
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
    },
    chartType: 'bar',
  },
};

export const MinimalWeights: Story = {
  render: (args) => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.05,
        budget: 0.05,
        age: 0.05,
        occasion: 0.85,
      },
      gift_features: {
        category: 0.1,
        price: 0.1,
        rating: 0.8,
      },
    },
    chartType: 'radar',
  },
};
