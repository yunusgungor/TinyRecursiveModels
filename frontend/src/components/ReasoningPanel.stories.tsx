import type { Meta, StoryObj } from '@storybook/react';
import { ReasoningPanel } from './ReasoningPanel';
import type { ReasoningTrace, GiftItem, UserProfile, ReasoningFilter } from '@/types/reasoning';
import { useState } from 'react';

const mockGift: GiftItem = {
  id: 'gift-123',
  name: 'Kahve Makinesi Deluxe',
  price: 1500,
  category: 'Ev & Yaşam',
  image_url: 'https://via.placeholder.com/300x300',
  rating: 4.5,
  availability: true,
  description: 'Profesyonel kahve deneyimi için ideal',
};

const mockUserProfile: UserProfile = {
  hobbies: ['kahve', 'yemek pişirme', 'teknoloji'],
  age: 32,
  budget: 2000,
  occasion: 'doğum günü',
  gender: 'erkek',
  relationship: 'arkadaş',
};

const mockReasoningTrace: ReasoningTrace = {
  tool_selection: [
    {
      name: 'review_analysis',
      selected: true,
      score: 0.85,
      reason: 'Yüksek rating ve pozitif yorumlar',
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
      reason: 'Popüler kategori ve artan trend',
      confidence: 0.75,
      priority: 2,
      factors: {
        trend_score: 0.7,
        popularity: 0.75,
      },
    },
    {
      name: 'inventory_check',
      selected: false,
      score: 0.45,
      reason: 'Stok durumu belirsiz',
      confidence: 0.4,
      priority: 3,
    },
  ],
  category_matching: [
    {
      category_name: 'Ev & Yaşam',
      score: 0.88,
      reasons: [
        'Kullanıcının kahve hobisi ile uyumlu',
        'Yaş grubuna uygun kategori',
        'Doğum günü hediyesi için ideal',
      ],
      feature_contributions: {
        hobby_match: 0.9,
        age_appropriateness: 0.85,
        occasion_fit: 0.9,
      },
    },
    {
      category_name: 'Teknoloji',
      score: 0.65,
      reasons: [
        'Teknoloji hobisi ile kısmen uyumlu',
        'Modern cihaz kategorisi',
      ],
      feature_contributions: {
        hobby_match: 0.7,
        age_appropriateness: 0.6,
      },
    },
    {
      category_name: 'Mutfak',
      score: 0.82,
      reasons: [
        'Yemek pişirme hobisi ile uyumlu',
        'Pratik kullanım',
      ],
      feature_contributions: {
        hobby_match: 0.85,
        occasion_fit: 0.8,
      },
    },
  ],
  attention_weights: {
    user_features: {
      hobbies: 0.35,
      budget: 0.25,
      age: 0.15,
      occasion: 0.25,
    },
    gift_features: {
      category: 0.4,
      price: 0.35,
      rating: 0.25,
    },
  },
  thinking_steps: [
    {
      step: 1,
      action: 'Kullanıcı profili analizi',
      result: 'Kahve ve yemek pişirme hobileri tespit edildi',
      insight: 'Mutfak ve ev kategorileri önceliklendirilmeli',
    },
    {
      step: 2,
      action: 'Bütçe kontrolü',
      result: '2000 TL bütçe ile uyumlu ürünler filtrelendi',
      insight: '1500 TL fiyat optimal aralıkta',
    },
    {
      step: 3,
      action: 'Kategori eşleştirme',
      result: 'Ev & Yaşam kategorisi %88 uyum gösterdi',
      insight: 'Yüksek kategori uyumu tespit edildi',
    },
    {
      step: 4,
      action: 'Tool sonuçları değerlendirme',
      result: 'Review ve trend analizi pozitif',
      insight: 'Güvenilir öneri için yeterli veri mevcut',
    },
  ],
  confidence_explanation: {
    score: 0.85,
    level: 'high',
    factors: {
      positive: [
        'Hobi uyumu çok yüksek',
        'Bütçe içinde optimal fiyat',
        'Yüksek rating ve pozitif yorumlar',
        'Popüler ve trend kategori',
      ],
      negative: [
        'Stok durumu belirsiz',
        'Alternatif ürünler de mevcut',
      ],
    },
  },
};

const meta: Meta<typeof ReasoningPanel> = {
  title: 'Components/ReasoningPanel',
  component: ReasoningPanel,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof ReasoningPanel>;

// Interactive wrapper component
const InteractiveWrapper = (args: any) => {
  const [isOpen, setIsOpen] = useState(true);
  const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>([
    'tool_selection',
    'category_matching',
    'attention_weights',
    'thinking_steps',
  ]);
  const [chartType, setChartType] = useState<'bar' | 'radar'>('bar');

  return (
    <div>
      <button
        onClick={() => setIsOpen(true)}
        className="rounded-lg bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
      >
        Detaylı Analiz Göster
      </button>

      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
        chartType={chartType}
        onChartTypeChange={setChartType}
      />
    </div>
  );
};

export const Default: Story = {
  render: (args) => <InteractiveWrapper {...args} />,
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const ToolSelectionOnly: Story = {
  render: (args) => {
    const [isOpen, setIsOpen] = useState(true);
    const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>(['tool_selection']);

    return (
      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
      />
    );
  },
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const CategoryMatchingOnly: Story = {
  render: (args) => {
    const [isOpen, setIsOpen] = useState(true);
    const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>(['category_matching']);

    return (
      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
      />
    );
  },
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const AttentionWeightsOnly: Story = {
  render: (args) => {
    const [isOpen, setIsOpen] = useState(true);
    const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>(['attention_weights']);

    return (
      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
      />
    );
  },
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const ThinkingStepsOnly: Story = {
  render: (args) => {
    const [isOpen, setIsOpen] = useState(true);
    const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>(['thinking_steps']);

    return (
      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
      />
    );
  },
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const EmptyFilters: Story = {
  render: (args) => {
    const [isOpen, setIsOpen] = useState(true);
    const [activeFilters, setActiveFilters] = useState<ReasoningFilter[]>([]);

    return (
      <ReasoningPanel
        {...args}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        activeFilters={activeFilters}
        onFilterChange={setActiveFilters}
      />
    );
  },
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
};

export const MobileView: Story = {
  render: (args) => <InteractiveWrapper {...args} />,
  args: {
    reasoningTrace: mockReasoningTrace,
    gift: mockGift,
    userProfile: mockUserProfile,
  },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
};
