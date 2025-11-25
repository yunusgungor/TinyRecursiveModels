import type { Meta, StoryObj } from '@storybook/react';
import { ComparisonView } from './ComparisonView';
import { EnhancedGiftRecommendation } from '@/types/reasoning';

/**
 * ComparisonView displays a side-by-side comparison of selected gift recommendations.
 * 
 * ## Features
 * - Side-by-side gift card display
 * - Category scores comparison chart
 * - Attention weights comparison
 * - Confidence score comparison table
 * - Responsive layout (stacks on mobile)
 * 
 * ## Accessibility
 * - ARIA labels for screen readers
 * - Keyboard navigable
 * - Semantic HTML structure
 * - Focus management
 * 
 * ## Usage
 * Used when users select multiple gifts for comparison. Typically accessed
 * from the recommendations page when 2 or more gifts are selected.
 */
const meta = {
  title: 'Components/ComparisonView',
  component: ComparisonView,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
  argTypes: {
    onExit: {
      action: 'exit comparison',
      description: 'Callback when user exits comparison mode',
    },
  },
} satisfies Meta<typeof ComparisonView>;

export default meta;
type Story = StoryObj<typeof meta>;

// Mock data for recommendations
const mockRecommendation1: EnhancedGiftRecommendation = {
  gift: {
    id: 'gift-1',
    name: 'Premium Kahve Makinesi',
    price: 1500,
    category: 'Ev & Yaşam',
    image_url: 'https://via.placeholder.com/400x400/3b82f6/ffffff?text=Kahve+Makinesi',
    rating: 4.5,
    availability: true,
  },
  reasoning: [
    'Kullanıcının kahve hobisi ile mükemmel eşleşme',
    'Bütçe aralığına uygun fiyat',
    'Yüksek rating ve pozitif yorumlar',
  ],
  confidence: 0.92,
  reasoning_trace: {
    tool_selection: [
      {
        name: 'review_analysis',
        selected: true,
        score: 0.85,
        reason: 'Yüksek rating',
        confidence: 0.9,
        priority: 1,
      },
    ],
    category_matching: [
      {
        category_name: 'Ev & Yaşam',
        score: 0.88,
        reasons: ['Kahve hobisi ile uyumlu'],
        feature_contributions: {
          hobby_match: 0.9,
          age_appropriateness: 0.85,
        },
      },
      {
        category_name: 'Mutfak',
        score: 0.82,
        reasons: ['Yemek pişirme hobisi ile uyumlu'],
        feature_contributions: {
          hobby_match: 0.85,
        },
      },
      {
        category_name: 'Teknoloji',
        score: 0.65,
        reasons: ['Modern cihaz'],
        feature_contributions: {
          hobby_match: 0.7,
        },
      },
    ],
    attention_weights: {
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
    thinking_steps: [],
  },
};

const mockRecommendation2: EnhancedGiftRecommendation = {
  gift: {
    id: 'gift-2',
    name: 'Profesyonel Espresso Makinesi',
    price: 2200,
    category: 'Ev & Yaşam',
    image_url: 'https://via.placeholder.com/400x400/10b981/ffffff?text=Espresso+Makinesi',
    rating: 4.8,
    availability: true,
  },
  reasoning: [
    'Üst segment kahve deneyimi',
    'Profesyonel özellikler',
    'Mükemmel kullanıcı değerlendirmeleri',
  ],
  confidence: 0.85,
  reasoning_trace: {
    tool_selection: [
      {
        name: 'review_analysis',
        selected: true,
        score: 0.92,
        reason: 'Çok yüksek rating',
        confidence: 0.95,
        priority: 1,
      },
    ],
    category_matching: [
      {
        category_name: 'Ev & Yaşam',
        score: 0.85,
        reasons: ['Kahve hobisi ile uyumlu'],
        feature_contributions: {
          hobby_match: 0.88,
          age_appropriateness: 0.82,
        },
      },
      {
        category_name: 'Mutfak',
        score: 0.78,
        reasons: ['Mutfak ekipmanı'],
        feature_contributions: {
          hobby_match: 0.8,
        },
      },
      {
        category_name: 'Teknoloji',
        score: 0.72,
        reasons: ['Gelişmiş teknoloji'],
        feature_contributions: {
          hobby_match: 0.75,
        },
      },
    ],
    attention_weights: {
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
    thinking_steps: [],
  },
};

const mockRecommendation3: EnhancedGiftRecommendation = {
  gift: {
    id: 'gift-3',
    name: 'Kompakt Kahve Makinesi',
    price: 899,
    category: 'Ev & Yaşam',
    image_url: 'https://via.placeholder.com/400x400/f59e0b/ffffff?text=Kompakt+Kahve',
    rating: 4.2,
    availability: true,
  },
  reasoning: [
    'Ekonomik seçenek',
    'Kompakt tasarım',
    'İyi değerlendirmeler',
  ],
  confidence: 0.68,
  reasoning_trace: {
    tool_selection: [
      {
        name: 'review_analysis',
        selected: true,
        score: 0.75,
        reason: 'İyi rating',
        confidence: 0.8,
        priority: 1,
      },
    ],
    category_matching: [
      {
        category_name: 'Ev & Yaşam',
        score: 0.75,
        reasons: ['Kahve hobisi ile uyumlu'],
        feature_contributions: {
          hobby_match: 0.78,
          age_appropriateness: 0.72,
        },
      },
      {
        category_name: 'Mutfak',
        score: 0.68,
        reasons: ['Mutfak ekipmanı'],
        feature_contributions: {
          hobby_match: 0.7,
        },
      },
      {
        category_name: 'Teknoloji',
        score: 0.55,
        reasons: ['Temel teknoloji'],
        feature_contributions: {
          hobby_match: 0.6,
        },
      },
    ],
    attention_weights: {
      user_features: {
        hobbies: 0.3,
        budget: 0.4,
        age: 0.2,
        occasion: 0.1,
      },
      gift_features: {
        category: 0.4,
        price: 0.4,
        rating: 0.2,
      },
    },
    thinking_steps: [],
  },
};

/**
 * Default comparison view with two gifts
 */
export const TwoGifts: Story = {
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2],
  },
};

/**
 * Comparison view with three gifts
 */
export const ThreeGifts: Story = {
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2, mockRecommendation3],
  },
};

/**
 * Comparison with high confidence gifts
 */
export const HighConfidenceComparison: Story = {
  args: {
    recommendations: [
      {
        ...mockRecommendation1,
        confidence: 0.95,
      },
      {
        ...mockRecommendation2,
        confidence: 0.92,
      },
    ],
  },
};

/**
 * Comparison with mixed confidence levels
 */
export const MixedConfidence: Story = {
  args: {
    recommendations: [
      {
        ...mockRecommendation1,
        confidence: 0.92,
      },
      {
        ...mockRecommendation2,
        confidence: 0.65,
      },
      {
        ...mockRecommendation3,
        confidence: 0.35,
      },
    ],
  },
};

/**
 * Comparison with different price ranges
 */
export const DifferentPriceRanges: Story = {
  args: {
    recommendations: [
      {
        ...mockRecommendation1,
        gift: {
          ...mockRecommendation1.gift,
          price: 500,
        },
      },
      {
        ...mockRecommendation2,
        gift: {
          ...mockRecommendation2.gift,
          price: 1500,
        },
      },
      {
        ...mockRecommendation3,
        gift: {
          ...mockRecommendation3.gift,
          price: 3000,
        },
      },
    ],
  },
};

/**
 * Comparison with similar category scores
 */
export const SimilarScores: Story = {
  args: {
    recommendations: [
      {
        ...mockRecommendation1,
        confidence: 0.85,
        reasoning_trace: {
          ...mockRecommendation1.reasoning_trace!,
          category_matching: mockRecommendation1.reasoning_trace!.category_matching.map(
            (cat) => ({
              ...cat,
              score: 0.8,
            })
          ),
        },
      },
      {
        ...mockRecommendation2,
        confidence: 0.83,
        reasoning_trace: {
          ...mockRecommendation2.reasoning_trace!,
          category_matching: mockRecommendation2.reasoning_trace!.category_matching.map(
            (cat) => ({
              ...cat,
              score: 0.78,
            })
          ),
        },
      },
    ],
  },
};

/**
 * Mobile viewport comparison
 */
export const MobileView: Story = {
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2],
  },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
};

/**
 * Tablet viewport comparison
 */
export const TabletView: Story = {
  args: {
    recommendations: [mockRecommendation1, mockRecommendation2, mockRecommendation3],
  },
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
};

/**
 * Empty state (should not happen in practice)
 */
export const EmptyState: Story = {
  args: {
    recommendations: [],
  },
};

/**
 * Single gift (edge case)
 */
export const SingleGift: Story = {
  args: {
    recommendations: [mockRecommendation1],
  },
};
