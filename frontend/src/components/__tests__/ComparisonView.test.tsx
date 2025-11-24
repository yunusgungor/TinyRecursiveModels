import { describe, test, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ComparisonView } from '../ComparisonView';
import { EnhancedGiftRecommendation } from '@/types/reasoning';

/**
 * Unit tests for ComparisonView component
 */

const mockRecommendations: EnhancedGiftRecommendation[] = [
  {
    gift: {
      id: '1',
      name: 'Premium Kahve Makinesi',
      price: 2499.99,
      image_url: 'https://example.com/image1.jpg',
      category: 'Ev & Yaşam',
      rating: 4.5,
      availability: true,
    },
    reasoning: [
      'Kullanıcının hobi listesinde kahve yapımı var',
      'Bütçe aralığına uygun fiyat',
    ],
    confidence: 0.85,
    reasoning_trace: {
      tool_selection: [],
      category_matching: [
        {
          category_name: 'Ev & Yaşam',
          score: 0.9,
          reasons: ['Hobi eşleşmesi', 'Yaş uygunluğu'],
          feature_contributions: { hobbies: 0.6, age: 0.3 },
        },
        {
          category_name: 'Elektronik',
          score: 0.7,
          reasons: ['Teknoloji ilgisi'],
          feature_contributions: { hobbies: 0.5 },
        },
      ],
      attention_weights: {
        user_features: { hobbies: 0.4, budget: 0.3, age: 0.2, occasion: 0.1 },
        gift_features: { category: 0.5, price: 0.3, rating: 0.2 },
      },
      thinking_steps: [],
    },
  },
  {
    gift: {
      id: '2',
      name: 'Yoga Matı',
      price: 299.99,
      image_url: 'https://example.com/image2.jpg',
      category: 'Spor',
      rating: 4.2,
      availability: true,
    },
    reasoning: [
      'Spor hobisine uygun',
      'Uygun fiyat aralığı',
    ],
    confidence: 0.75,
    reasoning_trace: {
      tool_selection: [],
      category_matching: [
        {
          category_name: 'Spor',
          score: 0.85,
          reasons: ['Hobi eşleşmesi'],
          feature_contributions: { hobbies: 0.7 },
        },
        {
          category_name: 'Sağlık',
          score: 0.6,
          reasons: ['Wellness ilgisi'],
          feature_contributions: { hobbies: 0.4 },
        },
      ],
      attention_weights: {
        user_features: { hobbies: 0.5, budget: 0.2, age: 0.2, occasion: 0.1 },
        gift_features: { category: 0.6, price: 0.2, rating: 0.2 },
      },
      thinking_steps: [],
    },
  },
];

describe('ComparisonView', () => {
  describe('Gift selection logic', () => {
    test('renders all selected gifts', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      // Verify both gifts are rendered
      expect(container.textContent).toContain('Premium Kahve Makinesi');
      expect(container.textContent).toContain('Yoga Matı');
    });

    test('displays correct number of gift cards', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const articles = container.querySelectorAll('[role="article"]');
      expect(articles.length).toBe(mockRecommendations.length);
    });

    test('renders with single gift', () => {
      const singleRecommendation = [mockRecommendations[0]];
      
      const { container } = render(
        <ComparisonView
          recommendations={singleRecommendation}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Premium Kahve Makinesi');
      const articles = container.querySelectorAll('[role="article"]');
      expect(articles.length).toBe(1);
    });

    test('renders with three gifts', () => {
      const threeRecommendations = [
        ...mockRecommendations,
        {
          ...mockRecommendations[0],
          gift: { ...mockRecommendations[0].gift, id: '3', name: 'Kitap Seti' },
        },
      ];
      
      const { container } = render(
        <ComparisonView
          recommendations={threeRecommendations}
          onExit={() => {}}
        />
      );

      const articles = container.querySelectorAll('[role="article"]');
      expect(articles.length).toBe(3);
    });
  });

  describe('Comparison view rendering', () => {
    test('displays comparison header', () => {
      render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(screen.getByText('Hediye Karşılaştırma')).toBeTruthy();
    });

    test('displays exit button', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const exitButton = container.querySelector('[aria-label="Karşılaştırma modundan çık"]');
      expect(exitButton).toBeTruthy();
      expect(exitButton?.textContent).toContain('Karşılaştırmayı Kapat');
    });

    test('calls onExit when exit button is clicked', async () => {
      const user = userEvent.setup();
      const handleExit = vi.fn();
      
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={handleExit}
        />
      );

      const exitButton = container.querySelector('[aria-label="Karşılaştırma modundan çık"]') as HTMLElement;
      await user.click(exitButton);

      expect(handleExit).toHaveBeenCalledTimes(1);
    });

    test('displays gift images', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const images = container.querySelectorAll('img');
      expect(images.length).toBeGreaterThanOrEqual(mockRecommendations.length);
    });

    test('displays gift prices', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      // Check for Turkish Lira formatting
      expect(container.textContent).toMatch(/₺/);
      expect(container.textContent).toMatch(/2\.499,99/);
      expect(container.textContent).toMatch(/299,99/);
    });

    test('displays gift categories', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Ev & Yaşam');
      expect(container.textContent).toContain('Spor');
    });

    test('displays confidence indicators', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const confidenceIndicators = container.querySelectorAll('[aria-label*="Güven skoru"]');
      expect(confidenceIndicators.length).toBeGreaterThanOrEqual(mockRecommendations.length);
    });

    test('displays reasoning for each gift', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Kullanıcının hobi listesinde kahve yapımı var');
      expect(container.textContent).toContain('Spor hobisine uygun');
    });
  });

  describe('Comparison charts', () => {
    test('displays category comparison section', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Kategori Skorları Karşılaştırması');
    });

    test('displays attention weights comparison section', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Attention Weights Karşılaştırması');
    });

    test('displays confidence comparison table', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      expect(container.textContent).toContain('Güven Skoru Karşılaştırması');
    });

    test('category comparison shows all categories', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      // Check for categories from both recommendations
      expect(container.textContent).toContain('Ev & Yaşam');
      expect(container.textContent).toContain('Elektronik');
      expect(container.textContent).toContain('Spor');
      expect(container.textContent).toContain('Sağlık');
    });

    test('confidence table shows all gifts', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      // Verify table structure
      const table = container.querySelector('table');
      expect(table).toBeTruthy();

      // Verify table headers
      expect(container.textContent).toContain('Hediye');
      expect(container.textContent).toContain('Güven Skoru');
      expect(container.textContent).toContain('Fiyat');
    });

    test('handles recommendations without reasoning trace', () => {
      const recommendationsWithoutTrace = mockRecommendations.map(rec => ({
        ...rec,
        reasoning_trace: undefined,
      }));

      const { container } = render(
        <ComparisonView
          recommendations={recommendationsWithoutTrace}
          onExit={() => {}}
        />
      );

      // Should still render basic comparison
      expect(container.textContent).toContain('Hediye Karşılaştırma');
      expect(container.textContent).toContain('Premium Kahve Makinesi');
    });

    test('handles recommendations with partial reasoning trace', () => {
      const partialRecommendations = [
        mockRecommendations[0],
        {
          ...mockRecommendations[1],
          reasoning_trace: undefined,
        },
      ];

      const { container } = render(
        <ComparisonView
          recommendations={partialRecommendations}
          onExit={() => {}}
        />
      );

      // Should render without errors
      expect(container.textContent).toContain('Hediye Karşılaştırma');
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const mainRegion = container.querySelector('[aria-label="Hediye karşılaştırma görünümü"]');
      expect(mainRegion).toBeTruthy();
      expect(mainRegion?.getAttribute('role')).toBe('region');
    });

    test('gift cards have proper ARIA labels', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const articles = container.querySelectorAll('[role="article"]');
      articles.forEach((article, idx) => {
        const ariaLabel = article.getAttribute('aria-label');
        expect(ariaLabel).toBeTruthy();
        expect(ariaLabel).toContain('Hediye');
        expect(ariaLabel).toContain(mockRecommendations[idx].gift.name);
      });
    });

    test('exit button has proper ARIA label', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const exitButton = container.querySelector('[aria-label="Karşılaştırma modundan çık"]');
      expect(exitButton).toBeTruthy();
    });
  });

  describe('Responsive layout', () => {
    test('uses grid layout for gift cards', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
        />
      );

      const gridContainer = container.querySelector('.grid');
      expect(gridContainer).toBeTruthy();
      expect(gridContainer?.className).toContain('grid-cols-1');
      expect(gridContainer?.className).toContain('md:grid-cols-2');
      expect(gridContainer?.className).toContain('lg:grid-cols-3');
    });
  });

  describe('Custom className', () => {
    test('applies custom className', () => {
      const { container } = render(
        <ComparisonView
          recommendations={mockRecommendations}
          onExit={() => {}}
          className="custom-test-class"
        />
      );

      const mainRegion = container.querySelector('[role="region"]');
      expect(mainRegion?.className).toContain('custom-test-class');
    });
  });
});
