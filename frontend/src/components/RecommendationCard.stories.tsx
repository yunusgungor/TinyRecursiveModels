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

const mockRecommendation = {
  gift: {
    id: '12345',
    name: 'Premium Coffee Set',
    category: 'Kitchen & Dining',
    price: 299.99,
    rating: 4.5,
    imageUrl: 'https://cdn.dummyimage.com/400x400/cccccc/000000.png&text=Coffee+Set',
    trendyolUrl: 'https://www.trendyol.com/product/12345',
    description: 'High-quality coffee set with grinder and accessories',
    tags: ['coffee', 'kitchen', 'gift'],
    ageSuitability: [25, 65] as [number, number],
    occasionFit: ['birthday', 'anniversary'],
    inStock: true,
  },
  confidenceScore: 0.92,
  reasoning: [
    'Matches user\'s cooking hobby',
    'Within budget range',
    'High rating and positive reviews',
    'Popular in the category',
  ],
  toolInsights: {
    priceMatch: 0.9,
    reviewScore: 0.85,
    trendScore: 0.88,
  },
  rank: 1,
};

const mockToolResults = {
  priceComparison: {
    bestPrice: 299.99,
    averagePrice: 350.0,
    priceRange: '250-400 TL',
    savingsPercentage: 14.3,
    checkedPlatforms: ['Trendyol', 'Hepsiburada', 'N11'],
  },
  reviewAnalysis: {
    averageRating: 4.5,
    totalReviews: 234,
    sentimentScore: 0.85,
    keyPositives: ['quality', 'design', 'value for money'],
    keyNegatives: ['shipping time'],
    recommendationConfidence: 0.92,
  },
  trendAnalysis: {
    trendDirection: 'rising',
    popularityScore: 0.88,
    growthRate: 15.5,
    trendingItems: ['coffee makers', 'espresso machines'],
  },
};

export const Default: Story = {
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults,
  },
};

export const HighConfidence: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      confidenceScore: 0.95,
    },
    toolResults: mockToolResults,
  },
};

export const MediumConfidence: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      confidenceScore: 0.65,
    },
    toolResults: mockToolResults,
  },
};

export const LowConfidence: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      confidenceScore: 0.35,
    },
    toolResults: mockToolResults,
  },
};

export const OutOfStock: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      gift: {
        ...mockRecommendation.gift,
        inStock: false,
      },
    },
    toolResults: mockToolResults,
  },
};

export const ExpensiveItem: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      gift: {
        ...mockRecommendation.gift,
        name: 'Professional Espresso Machine',
        price: 2499.99,
      },
    },
    toolResults: mockToolResults,
  },
};

export const LowRating: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      gift: {
        ...mockRecommendation.gift,
        rating: 2.5,
      },
    },
    toolResults: {
      ...mockToolResults,
      reviewAnalysis: {
        ...mockToolResults.reviewAnalysis!,
        averageRating: 2.5,
        sentimentScore: 0.45,
        keyNegatives: ['poor quality', 'breaks easily', 'not worth the price'],
      },
    },
  },
};

export const MinimalToolResults: Story = {
  args: {
    recommendation: mockRecommendation,
    toolResults: {},
  },
};
