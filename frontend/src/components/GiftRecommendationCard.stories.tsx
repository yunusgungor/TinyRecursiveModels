import type { Meta, StoryObj } from '@storybook/react';
import { GiftRecommendationCard } from './GiftRecommendationCard';
import { EnhancedGiftRecommendation } from '@/types/reasoning';

const meta = {
  title: 'Components/GiftRecommendationCard',
  component: GiftRecommendationCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    onShowDetails: { action: 'show details clicked' },
    onSelect: { action: 'selection toggled' },
  },
} satisfies Meta<typeof GiftRecommendationCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockRecommendation: EnhancedGiftRecommendation = {
  gift: {
    id: '1',
    name: 'Premium Kahve Makinesi',
    price: 2499.99,
    image_url: 'https://via.placeholder.com/400',
    category: 'Ev & Yaşam',
    rating: 4.5,
    availability: true,
  },
  reasoning: [
    'Kullanıcının hobi listesinde kahve yapımı var, bu ürün mükemmel bir eşleşme',
    'Bütçe aralığına uygun fiyat',
    'Yaş grubuna uygun sofistike bir hediye',
  ],
  confidence: 0.85,
};

const mockToolResults = {
  review_analysis: {
    average_rating: 4.5,
    review_count: 1234,
  },
  trend_analysis: {
    trending: true,
    trend_score: 0.8,
  },
  inventory_check: {
    available: true,
    stock_count: 50,
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
      confidence: 0.92,
    },
    toolResults: mockToolResults,
  },
};

export const MediumConfidence: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      confidence: 0.65,
    },
    toolResults: mockToolResults,
  },
};

export const LowConfidence: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      confidence: 0.35,
    },
    toolResults: mockToolResults,
  },
};

export const LongReasoning: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      reasoning: [
        'Kullanıcının hobi listesinde kahve yapımı var, bu ürün mükemmel bir eşleşme',
        'Bütçe aralığına uygun fiyat, kullanıcının belirlediği maksimum bütçenin altında',
        'Yaş grubuna uygun sofistike bir hediye, 30-40 yaş arası kullanıcılar için ideal',
        'Yüksek kaliteli malzeme ve dayanıklılık özellikleri',
        'Kolay kullanım ve temizlik imkanı',
        'Enerji tasarruflu model',
      ],
    },
    toolResults: mockToolResults,
  },
};

export const WithoutToolResults: Story = {
  args: {
    recommendation: mockRecommendation,
  },
};

export const Selected: Story = {
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults,
    isSelected: true,
  },
};

export const WithSelectionCheckbox: Story = {
  args: {
    recommendation: mockRecommendation,
    toolResults: mockToolResults,
    isSelected: false,
  },
};

export const MinimalReasoning: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      reasoning: ['Hobi eşleşmesi mükemmel'],
    },
    toolResults: mockToolResults,
  },
};

export const NoImage: Story = {
  args: {
    recommendation: {
      ...mockRecommendation,
      gift: {
        ...mockRecommendation.gift,
        image_url: undefined,
      },
    },
    toolResults: mockToolResults,
  },
};

export const PartialToolResults: Story = {
  args: {
    recommendation: mockRecommendation,
    toolResults: {
      review_analysis: {
        average_rating: 4.2,
        review_count: 500,
      },
    },
  },
};
