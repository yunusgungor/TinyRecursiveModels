import type { Meta, StoryObj } from '@storybook/react';
import { RecommendationCard } from './RecommendationCard';

const meta = {
  title: 'Components/RecommendationCard',
  component: RecommendationCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    onDetailsClick: { action: 'details clicked' },
    onTrendyolClick: { action: 'trendyol clicked' },
  },
} satisfies Meta<typeof RecommendationCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockGift = {
  id: '12345',
  name: 'Premium Coffee Set',
  category: 'Kitchen & Dining',
  price: 299.99,
  rating: 4.5,
  image_url: 'https://cdn.dummyimage.com/400x400/cccccc/000000.png&text=Coffee+Set',
  trendyol_url: 'https://www.trendyol.com/product/12345',
  description: 'High-quality coffee set with grinder and accessories',
  tags: ['coffee', 'kitchen', 'gift'],
  age_suitability: [25, 65] as [number, number],
  occasion_fit: ['birthday', 'anniversary'],
  in_stock: true,
};

const mockToolResults = {
  priceComparison: {
    best_price: 299.99,
    average_price: 350.0,
    price_range: '250-400 TL',
    savings_percentage: 14.3,
    checked_platforms: ['Trendyol', 'Hepsiburada', 'N11'],
  },
  reviewAnalysis: {
    average_rating: 4.5,
    total_reviews: 234,
    sentiment_score: 0.85,
    key_positives: ['quality', 'design', 'value for money'],
    key_negatives: ['shipping time'],
    recommendation_confidence: 0.92,
  },
  trendAnalysis: {
    trend_direction: 'rising',
    popularity_score: 0.88,
    growth_rate: 15.5,
    trending_items: ['coffee makers', 'espresso machines'],
  },
};

export const Default: Story = {
  args: {
    gift: mockGift,
    toolResults: mockToolResults,
    confidenceScore: 0.92,
    reasoning: [
      'Matches user\'s cooking hobby',
      'Within budget range',
      'High rating and positive reviews',
      'Popular in the category',
    ],
    rank: 1,
  },
};

export const HighConfidence: Story = {
  args: {
    ...Default.args,
    confidenceScore: 0.95,
  },
};

export const MediumConfidence: Story = {
  args: {
    ...Default.args,
    confidenceScore: 0.65,
  },
};

export const LowConfidence: Story = {
  args: {
    ...Default.args,
    confidenceScore: 0.35,
  },
};

export const OutOfStock: Story = {
  args: {
    ...Default.args,
    gift: {
      ...mockGift,
      in_stock: false,
    },
  },
};

export const ExpensiveItem: Story = {
  args: {
    ...Default.args,
    gift: {
      ...mockGift,
      name: 'Professional Espresso Machine',
      price: 2499.99,
    },
  },
};

export const LowRating: Story = {
  args: {
    ...Default.args,
    gift: {
      ...mockGift,
      rating: 2.5,
    },
    toolResults: {
      ...mockToolResults,
      reviewAnalysis: {
        ...mockToolResults.reviewAnalysis!,
        average_rating: 2.5,
        sentiment_score: 0.45,
        key_negatives: ['poor quality', 'breaks easily', 'not worth the price'],
      },
    },
  },
};

export const MinimalToolResults: Story = {
  args: {
    ...Default.args,
    toolResults: {},
  },
};
