import type { Meta, StoryObj } from '@storybook/react';
import { ToolSelectionCard } from './ToolSelectionCard';
import { ToolSelectionReasoning } from '@/types/reasoning';

const meta = {
  title: 'Components/ToolSelectionCard',
  component: ToolSelectionCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ToolSelectionCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const sampleToolSelection: ToolSelectionReasoning[] = [
  {
    name: 'review_analysis',
    selected: true,
    score: 0.85,
    reason: 'Yüksek rating eşleşmesi ve pozitif yorumlar',
    confidence: 0.9,
    priority: 1,
    factors: {
      rating_match: 0.9,
      review_sentiment: 0.85,
      review_count: 0.8,
    },
  },
  {
    name: 'trend_analysis',
    selected: true,
    score: 0.72,
    reason: 'Popüler trend kategorisinde',
    confidence: 0.75,
    priority: 2,
    factors: {
      trend_score: 0.8,
      popularity: 0.7,
    },
  },
  {
    name: 'inventory_check',
    selected: false,
    score: 0.45,
    reason: 'Stok durumu belirsiz',
    confidence: 0.4,
    priority: 3,
    factors: {
      availability: 0.5,
      stock_level: 0.4,
    },
  },
  {
    name: 'price_comparison',
    selected: true,
    score: 0.88,
    reason: 'Bütçe ile mükemmel uyum',
    confidence: 0.92,
    priority: 4,
    factors: {
      budget_match: 0.95,
      price_competitiveness: 0.85,
    },
  },
];

export const Default: Story = {
  args: {
    toolSelection: sampleToolSelection,
  },
};

export const AllSelected: Story = {
  args: {
    toolSelection: sampleToolSelection.map((tool) => ({
      ...tool,
      selected: true,
      confidence: 0.85,
    })),
  },
};

export const AllUnselected: Story = {
  args: {
    toolSelection: sampleToolSelection.map((tool) => ({
      ...tool,
      selected: false,
      confidence: 0.6,
    })),
  },
};

export const LowConfidence: Story = {
  args: {
    toolSelection: [
      {
        name: 'review_analysis',
        selected: true,
        score: 0.35,
        reason: 'Sınırlı yorum verisi',
        confidence: 0.3,
        priority: 1,
        factors: {
          review_count: 0.2,
        },
      },
      {
        name: 'trend_analysis',
        selected: false,
        score: 0.25,
        reason: 'Trend verisi yetersiz',
        confidence: 0.2,
        priority: 2,
      },
    ],
  },
};

export const NoFactors: Story = {
  args: {
    toolSelection: [
      {
        name: 'review_analysis',
        selected: true,
        score: 0.75,
        reason: 'Genel değerlendirme olumlu',
        confidence: 0.8,
        priority: 1,
      },
      {
        name: 'inventory_check',
        selected: false,
        score: 0.5,
        reason: 'Stok bilgisi alınamadı',
        confidence: 0.5,
        priority: 2,
      },
    ],
  },
};

export const Empty: Story = {
  args: {
    toolSelection: [],
  },
};

export const SingleTool: Story = {
  args: {
    toolSelection: [
      {
        name: 'price_comparison',
        selected: true,
        score: 0.95,
        reason: 'Mükemmel fiyat uyumu',
        confidence: 0.98,
        priority: 1,
        factors: {
          budget_match: 1.0,
          price_competitiveness: 0.9,
        },
      },
    ],
  },
};
