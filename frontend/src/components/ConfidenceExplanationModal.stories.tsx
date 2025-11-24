import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { ConfidenceExplanationModal } from './ConfidenceExplanationModal';
import { ConfidenceExplanation } from '@/types';

const meta = {
  title: 'Components/ConfidenceExplanationModal',
  component: ConfidenceExplanationModal,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ConfidenceExplanationModal>;

export default meta;
type Story = StoryObj<typeof meta>;

// Wrapper component to handle modal state
const ModalWrapper = ({ explanation }: { explanation: ConfidenceExplanation }) => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div>
      <button
        onClick={() => setIsOpen(true)}
        className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
      >
        Open Modal
      </button>
      <ConfidenceExplanationModal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        explanation={explanation}
      />
    </div>
  );
};

export const HighConfidence: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.92,
        level: 'high',
        factors: {
          positive: [
            'Hobi ile mükemmel eşleşme: Kullanıcının yemek pişirme hobisi ile tam uyumlu',
            'Bütçe içinde: Belirlenen bütçenin %85\'i kadar',
            'Yüksek değerlendirme: 4.8/5.0 yıldız',
            'Popüler ürün: Son 30 günde 500+ satış',
          ],
          negative: [
            'Stok durumu: Sadece 3 adet kaldı',
          ],
        },
      }}
    />
  ),
};

export const MediumConfidence: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.65,
        level: 'medium',
        factors: {
          positive: [
            'Yaş grubu uygun: 25-35 yaş arası için ideal',
            'Kategori eşleşmesi: Teknoloji kategorisi ile uyumlu',
          ],
          negative: [
            'Bütçe üstü: Belirlenen bütçenin %120\'si',
            'Orta değerlendirme: 3.5/5.0 yıldız',
            'Hobi eşleşmesi zayıf: Kullanıcı hobisi ile kısmi uyum',
          ],
        },
      }}
    />
  ),
};

export const LowConfidence: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.35,
        level: 'low',
        factors: {
          positive: [
            'Stokta mevcut: Hemen teslim edilebilir',
          ],
          negative: [
            'Hobi uyumsuzluğu: Kullanıcı hobisi ile eşleşmiyor',
            'Bütçe dışı: Belirlenen bütçenin %180\'i',
            'Düşük değerlendirme: 2.8/5.0 yıldız',
            'Yaş grubu uyumsuz: Farklı yaş grubu için tasarlanmış',
            'Kategori eşleşmesi yok: İlgi alanları ile uyuşmuyor',
          ],
        },
      }}
    />
  ),
};

export const OnlyPositiveFactors: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.95,
        level: 'high',
        factors: {
          positive: [
            'Mükemmel hobi eşleşmesi',
            'Bütçe içinde',
            'Yüksek değerlendirme',
            'Stokta bol miktarda',
            'Hızlı teslimat',
          ],
          negative: [],
        },
      }}
    />
  ),
};

export const OnlyNegativeFactors: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.15,
        level: 'low',
        factors: {
          positive: [],
          negative: [
            'Hobi ile hiç eşleşmiyor',
            'Bütçe çok üstünde',
            'Çok düşük değerlendirme',
            'Stokta yok',
            'Yaş grubu uyumsuz',
          ],
        },
      }}
    />
  ),
};

export const MinimalFactors: Story = {
  render: () => (
    <ModalWrapper
      explanation={{
        score: 0.70,
        level: 'medium',
        factors: {
          positive: ['Bütçe uygun'],
          negative: ['Stok az'],
        },
      }}
    />
  ),
};
