import type { Meta, StoryObj } from '@storybook/react';
import { ThinkingStepsTimeline } from './ThinkingStepsTimeline';
import { ThinkingStep } from '@/types/reasoning';

const meta = {
  title: 'Components/ThinkingStepsTimeline',
  component: ThinkingStepsTimeline,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    steps: {
      description: 'Array of thinking steps to display in chronological order',
    },
    onStepClick: {
      description: 'Callback function when a step is clicked',
      action: 'step clicked',
    },
    className: {
      description: 'Additional CSS classes',
    },
  },
} satisfies Meta<typeof ThinkingStepsTimeline>;

export default meta;
type Story = StoryObj<typeof meta>;

// Sample thinking steps data
const sampleSteps: ThinkingStep[] = [
  {
    step: 1,
    action: 'Kullanıcı profilini analiz et',
    result: 'Hobiler ve ilgi alanları belirlendi',
    insight: 'Kullanıcı açık hava aktivitelerini ve sporu tercih ediyor',
  },
  {
    step: 2,
    action: 'Kategori filtreleme uygula',
    result: '5 uygun kategori seçildi',
    insight: 'Spor ve açık hava kategorileri kullanıcı profiliyle eşleşiyor',
  },
  {
    step: 3,
    action: 'Bütçe optimizasyonu yap',
    result: 'Fiyat aralığı 500-1500 TL olarak belirlendi',
    insight: 'Kullanıcının bütçesi orta-üst segment ürünlere uygun',
  },
  {
    step: 4,
    action: 'Tool sonuçlarını birleştir',
    result: 'Yorum analizi ve trend verileri entegre edildi',
    insight: 'Yüksek puanlı ve trend olan ürünler önceliklendirildi',
  },
  {
    step: 5,
    action: 'Final sıralama ve öneriler',
    result: '10 hediye önerisi oluşturuldu',
    insight: 'Öneriler güven skoruna göre sıralandı',
  },
];

const longSteps: ThinkingStep[] = [
  ...sampleSteps,
  {
    step: 6,
    action: 'Stok kontrolü yap',
    result: 'Tüm öneriler stokta mevcut',
    insight: 'Teslimat süresi 2-3 gün',
  },
  {
    step: 7,
    action: 'Fiyat karşılaştırması',
    result: 'Rekabetçi fiyatlar belirlendi',
    insight: 'Önerilen ürünler piyasa ortalamasının altında',
  },
  {
    step: 8,
    action: 'Kullanıcı geçmişi analizi',
    result: 'Önceki satın alımlar incelendi',
    insight: 'Kullanıcı kaliteli ürünleri tercih ediyor',
  },
  {
    step: 9,
    action: 'Sezon uygunluğu kontrolü',
    result: 'Mevsimsel ürünler filtrelendi',
    insight: 'Kış sporları ekipmanları öne çıkarıldı',
  },
  {
    step: 10,
    action: 'Final doğrulama',
    result: 'Tüm kriterler karşılandı',
    insight: 'Öneriler kullanıcıya sunulmaya hazır',
  },
];

const singleStep: ThinkingStep[] = [
  {
    step: 1,
    action: 'Hızlı öneri oluştur',
    result: 'Basit filtreleme uygulandı',
    insight: 'Temel kriterler kullanıldı',
  },
];

// Default story with sample steps
export const Default: Story = {
  args: {
    steps: sampleSteps,
  },
};

// Story with a single step
export const SingleStep: Story = {
  args: {
    steps: singleStep,
  },
};

// Story with many steps (scrollable)
export const LongTimeline: Story = {
  args: {
    steps: longSteps,
  },
};

// Story with empty steps
export const EmptySteps: Story = {
  args: {
    steps: [],
  },
};

// Story with custom click handler
export const WithClickHandler: Story = {
  args: {
    steps: sampleSteps,
    onStepClick: (step) => {
      console.log('Clicked step:', step);
      alert(`Adım ${step.step} tıklandı: ${step.action}`);
    },
  },
};

// Story demonstrating keyboard navigation
export const KeyboardNavigation: Story = {
  args: {
    steps: sampleSteps,
  },
  parameters: {
    docs: {
      description: {
        story: 'Tab tuşu ile adımlar arasında gezinin, Enter veya Space ile detayları açın/kapatın.',
      },
    },
  },
};

// Story with out-of-order steps (should auto-sort)
export const OutOfOrderSteps: Story = {
  args: {
    steps: [
      {
        step: 3,
        action: 'Üçüncü adım',
        result: 'Sonuç 3',
        insight: 'İçgörü 3',
      },
      {
        step: 1,
        action: 'Birinci adım',
        result: 'Sonuç 1',
        insight: 'İçgörü 1',
      },
      {
        step: 2,
        action: 'İkinci adım',
        result: 'Sonuç 2',
        insight: 'İçgörü 2',
      },
    ],
  },
  parameters: {
    docs: {
      description: {
        story: 'Adımlar otomatik olarak kronolojik sıraya göre düzenlenir.',
      },
    },
  },
};

// Story with very long text
export const LongTextContent: Story = {
  args: {
    steps: [
      {
        step: 1,
        action: 'Çok uzun bir aksiyon açıklaması ile detaylı analiz süreci başlatılıyor ve tüm parametreler kontrol ediliyor',
        result: 'Uzun bir sonuç metni: Kullanıcı profili detaylı olarak incelendi, tüm hobiler, ilgi alanları, yaş grubu, bütçe kısıtlamaları ve özel tercihler dikkate alındı. Sistem 150 farklı kategoriyi taradı ve en uygun 5 tanesini belirledi.',
        insight: 'Detaylı içgörü: Kullanıcının geçmiş satın alma davranışları, favori kategorileri, fiyat hassasiyeti ve marka tercihleri analiz edildiğinde, açık hava sporları ve teknoloji ürünlerine yönelik güçlü bir eğilim gözlemlendi. Bu bilgiler ışığında öneriler optimize edildi.',
      },
      {
        step: 2,
        action: 'İkinci adım',
        result: 'Normal uzunlukta sonuç',
        insight: 'Standart içgörü',
      },
    ],
  },
};

// Story with custom styling
export const CustomStyling: Story = {
  args: {
    steps: sampleSteps,
    className: 'max-w-2xl mx-auto shadow-xl',
  },
};
