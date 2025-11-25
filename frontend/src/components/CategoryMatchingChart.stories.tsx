import type { Meta, StoryObj } from '@storybook/react';
import { CategoryMatchingChart } from './CategoryMatchingChart';
import { CategoryMatchingReasoning } from '@/types/reasoning';

/**
 * CategoryMatchingChart displays category matching scores as horizontal bar charts.
 * 
 * ## Color Coding
 * - **Green** (>0.7): High match score
 * - **Yellow** (0.3-0.7): Medium match score
 * - **Red** (<0.3): Low match score
 * 
 * ## Features
 * - Shows at least top 3 categories
 * - Scores displayed as percentages
 * - Click to expand reasons
 * - Feature contributions breakdown
 * - Responsive layout
 * 
 * ## Accessibility
 * - ARIA labels for chart elements
 * - Keyboard navigable
 * - Screen reader friendly
 * 
 * ## Usage
 * Used in ReasoningPanel to show category matching reasoning.
 */
const meta = {
  title: 'Components/CategoryMatchingChart',
  component: CategoryMatchingChart,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    categories: {
      description: 'Array of category matching data with scores and reasons',
    },
    onCategoryClick: {
      description: 'Optional callback when a category is clicked',
      action: 'category clicked',
    },
  },
} satisfies Meta<typeof CategoryMatchingChart>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockCategories: CategoryMatchingReasoning[] = [
  {
    category_name: 'Elektronik',
    score: 0.85,
    reasons: [
      'Hobi eşleşmesi: Teknoloji meraklısı',
      'Yaş uygunluğu: Genç yetişkin',
      'Bütçe uygunluğu: Orta-yüksek bütçe',
    ],
    feature_contributions: {
      hobby: 0.9,
      age: 0.8,
      budget: 0.85,
    },
  },
  {
    category_name: 'Kitap',
    score: 0.65,
    reasons: [
      'Hobi eşleşmesi: Okuma sevgisi',
      'Yaş uygunluğu: Her yaş',
    ],
    feature_contributions: {
      hobby: 0.7,
      age: 0.6,
    },
  },
  {
    category_name: 'Spor Malzemeleri',
    score: 0.45,
    reasons: [
      'Hobi eşleşmesi: Aktif yaşam',
    ],
    feature_contributions: {
      hobby: 0.5,
    },
  },
  {
    category_name: 'Ev Dekorasyonu',
    score: 0.25,
    reasons: [
      'Düşük ilgi: Ev dekorasyonuna ilgi yok',
    ],
    feature_contributions: {
      hobby: 0.2,
      occasion: 0.3,
    },
  },
];

/**
 * Default category matching with mixed scores
 * Shows green, yellow, and red bars based on score ranges
 */
export const Default: Story = {
  args: {
    categories: mockCategories,
  },
};

/**
 * All categories with high scores (>0.7)
 * All bars should be green
 */
export const HighScoresOnly: Story = {
  args: {
    categories: [
      {
        category_name: 'Teknoloji',
        score: 0.92,
        reasons: ['Mükemmel hobi eşleşmesi', 'Yüksek bütçe uygunluğu'],
        feature_contributions: {
          hobby: 0.95,
          budget: 0.9,
        },
      },
      {
        category_name: 'Oyun',
        score: 0.88,
        reasons: ['Güçlü hobi eşleşmesi', 'Yaş uygunluğu'],
        feature_contributions: {
          hobby: 0.9,
          age: 0.85,
        },
      },
      {
        category_name: 'Elektronik Aksesuarlar',
        score: 0.75,
        reasons: ['İyi hobi eşleşmesi'],
        feature_contributions: {
          hobby: 0.8,
        },
      },
    ],
  },
};

/**
 * All categories with low scores (<0.3)
 * All bars should be red
 */
export const LowScoresOnly: Story = {
  args: {
    categories: [
      {
        category_name: 'Bahçe',
        score: 0.28,
        reasons: ['Düşük ilgi'],
        feature_contributions: {
          hobby: 0.2,
        },
      },
      {
        category_name: 'Bebek Ürünleri',
        score: 0.15,
        reasons: ['İlgisiz kategori'],
        feature_contributions: {
          occasion: 0.1,
        },
      },
      {
        category_name: 'Otomotiv',
        score: 0.22,
        reasons: ['Düşük eşleşme'],
        feature_contributions: {
          hobby: 0.25,
        },
      },
    ],
  },
};

/**
 * Mixed score ranges demonstrating all color codes
 */
export const MixedScores: Story = {
  args: {
    categories: mockCategories,
  },
};

/**
 * Minimum of 3 categories (requirement validation)
 */
export const MinimumCategories: Story = {
  args: {
    categories: [
      {
        category_name: 'Elektronik',
        score: 0.85,
        reasons: ['Hobi eşleşmesi'],
        feature_contributions: {
          hobby: 0.9,
        },
      },
      {
        category_name: 'Kitap',
        score: 0.65,
        reasons: ['Yaş uygunluğu'],
        feature_contributions: {
          age: 0.7,
        },
      },
      {
        category_name: 'Spor',
        score: 0.45,
        reasons: ['Orta eşleşme'],
        feature_contributions: {
          hobby: 0.5,
        },
      },
    ],
  },
};

/**
 * Many categories (8+) to test scrolling behavior
 */
export const ManyCategories: Story = {
  args: {
    categories: [
      {
        category_name: 'Elektronik',
        score: 0.95,
        reasons: ['Mükemmel eşleşme'],
        feature_contributions: { hobby: 0.95 },
      },
      {
        category_name: 'Kitap',
        score: 0.85,
        reasons: ['Çok iyi eşleşme'],
        feature_contributions: { hobby: 0.85 },
      },
      {
        category_name: 'Oyun',
        score: 0.75,
        reasons: ['İyi eşleşme'],
        feature_contributions: { hobby: 0.75 },
      },
      {
        category_name: 'Spor',
        score: 0.65,
        reasons: ['Orta eşleşme'],
        feature_contributions: { hobby: 0.65 },
      },
      {
        category_name: 'Müzik',
        score: 0.55,
        reasons: ['Orta-düşük eşleşme'],
        feature_contributions: { hobby: 0.55 },
      },
      {
        category_name: 'Moda',
        score: 0.45,
        reasons: ['Düşük eşleşme'],
        feature_contributions: { hobby: 0.45 },
      },
      {
        category_name: 'Ev',
        score: 0.35,
        reasons: ['Çok düşük eşleşme'],
        feature_contributions: { hobby: 0.35 },
      },
      {
        category_name: 'Bahçe',
        score: 0.25,
        reasons: ['Minimal eşleşme'],
        feature_contributions: { hobby: 0.25 },
      },
    ],
  },
};

/**
 * Empty state - no categories available
 */
export const EmptyState: Story = {
  args: {
    categories: [],
  },
};

/**
 * Interactive category click handler
 * Click on a category to see its details
 */
export const WithClickHandler: Story = {
  args: {
    categories: mockCategories,
    onCategoryClick: (category) => {
      console.log('Category clicked:', category);
      alert(`Kategori tıklandı: ${category.category_name} (${(category.score * 100).toFixed(0)}%)`);
    },
  },
};

export const NoFeatureContributions: Story = {
  args: {
    categories: [
      {
        category_name: 'Elektronik',
        score: 0.85,
        reasons: ['Hobi eşleşmesi'],
        feature_contributions: {},
      },
      {
        category_name: 'Kitap',
        score: 0.65,
        reasons: ['Yaş uygunluğu'],
        feature_contributions: {},
      },
      {
        category_name: 'Spor',
        score: 0.45,
        reasons: ['Orta eşleşme'],
        feature_contributions: {},
      },
    ],
  },
};

export const SingleReason: Story = {
  args: {
    categories: [
      {
        category_name: 'Elektronik',
        score: 0.85,
        reasons: ['Tek neden: Hobi eşleşmesi'],
        feature_contributions: {
          hobby: 0.9,
        },
      },
    ],
  },
};

export const MultipleReasons: Story = {
  args: {
    categories: [
      {
        category_name: 'Elektronik',
        score: 0.85,
        reasons: [
          'Hobi eşleşmesi: Teknoloji meraklısı',
          'Yaş uygunluğu: Genç yetişkin',
          'Bütçe uygunluğu: Orta-yüksek bütçe',
          'Occasion uygunluğu: Doğum günü hediyesi',
          'Trend analizi: Popüler kategori',
        ],
        feature_contributions: {
          hobby: 0.9,
          age: 0.8,
          budget: 0.85,
          occasion: 0.75,
          trend: 0.7,
        },
      },
    ],
  },
};
