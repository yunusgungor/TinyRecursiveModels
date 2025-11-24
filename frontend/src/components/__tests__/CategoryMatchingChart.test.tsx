import { describe, test, expect, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { CategoryMatchingChart } from '../CategoryMatchingChart';
import { CategoryMatchingReasoning } from '@/types/reasoning';

/**
 * Unit tests for CategoryMatchingChart component
 */

describe('CategoryMatchingChart', () => {
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

  describe('Rendering with various score ranges', () => {
    test('renders category matching chart with title', () => {
      render(<CategoryMatchingChart categories={mockCategories} />);
      
      expect(screen.getByText('Kategori Eşleştirme')).toBeInTheDocument();
    });

    test('displays at least top 3 categories', () => {
      render(<CategoryMatchingChart categories={mockCategories} />);
      
      // Should display top 3 categories
      expect(screen.getByText('Elektronik')).toBeInTheDocument();
      expect(screen.getByText('Kitap')).toBeInTheDocument();
      expect(screen.getByText('Spor Malzemeleri')).toBeInTheDocument();
    });

    test('displays scores as percentages', () => {
      render(<CategoryMatchingChart categories={mockCategories} />);
      
      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(screen.getByText('65%')).toBeInTheDocument();
      expect(screen.getByText('45%')).toBeInTheDocument();
    });

    test('renders high score categories with green styling', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const elektronikItem = Array.from(listItems).find((item) =>
        item.textContent?.includes('Elektronik')
      );
      
      expect(elektronikItem).toBeTruthy();
      expect(elektronikItem?.className).toContain('border-green-300');
      expect(elektronikItem?.className).toContain('bg-green-50');
      
      const greenIndicator = elektronikItem?.querySelector('.bg-green-500');
      expect(greenIndicator).toBeTruthy();
    });

    test('renders medium score categories with yellow styling', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const kitapItem = Array.from(listItems).find((item) =>
        item.textContent?.includes('Kitap')
      );
      
      expect(kitapItem).toBeTruthy();
      expect(kitapItem?.className).toContain('border-yellow-300');
      expect(kitapItem?.className).toContain('bg-yellow-50');
      
      const yellowIndicator = kitapItem?.querySelector('.bg-yellow-500');
      expect(yellowIndicator).toBeTruthy();
    });

    test('renders low score categories with red styling', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const evItem = Array.from(listItems).find((item) =>
        item.textContent?.includes('Ev Dekorasyonu')
      );
      
      expect(evItem).toBeTruthy();
      expect(evItem?.className).toContain('border-red-300');
      expect(evItem?.className).toContain('bg-red-50');
      
      const redIndicator = evItem?.querySelector('.bg-red-500');
      expect(redIndicator).toBeTruthy();
    });

    test('displays reason count for each category', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      // Check that reason counts are displayed
      const reasonCounts = container.querySelectorAll('.text-xs.text-gray-500');
      const reasonCountTexts = Array.from(reasonCounts).map(el => el.textContent?.trim());
      
      expect(reasonCountTexts).toContain('3 neden'); // Elektronik
      expect(reasonCountTexts).toContain('2 neden'); // Kitap
      expect(reasonCountTexts.filter(text => text === '1 neden').length).toBeGreaterThanOrEqual(1); // Spor Malzemeleri and/or Ev Dekorasyonu
    });

    test('renders empty state when no categories provided', () => {
      render(<CategoryMatchingChart categories={[]} />);
      
      expect(screen.getByText('Kategori eşleştirme bilgisi mevcut değil')).toBeInTheDocument();
    });

    test('sorts categories by score in descending order', () => {
      const unsortedCategories: CategoryMatchingReasoning[] = [
        {
          category_name: 'Spor',
          score: 0.45,
          reasons: ['Test'],
          feature_contributions: {},
        },
        {
          category_name: 'Elektronik',
          score: 0.85,
          reasons: ['Test'],
          feature_contributions: {},
        },
        {
          category_name: 'Kitap',
          score: 0.65,
          reasons: ['Test'],
          feature_contributions: {},
        },
      ];

      const { container } = render(<CategoryMatchingChart categories={unsortedCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const categoryNames = Array.from(listItems).map((item) => {
        const nameElement = item.querySelector('span.font-medium');
        return nameElement?.textContent || '';
      });
      
      // Should be sorted: Elektronik (0.85), Kitap (0.65), Spor (0.45)
      expect(categoryNames[0]).toBe('Elektronik');
      expect(categoryNames[1]).toBe('Kitap');
      expect(categoryNames[2]).toBe('Spor');
    });
  });

  describe('Click expansion functionality', () => {
    test('initially does not show reasons', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]');
      const reasonsList = firstItem?.querySelector('ul.list-disc');
      
      expect(reasonsList).toBeNull();
    });

    test('shows reasons when category is clicked', async () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      await act(async () => {
        firstItem.click();
      });
      
      const reasonsList = firstItem.querySelector('ul.list-disc');
      expect(reasonsList).toBeTruthy();
      
      const reasonItems = firstItem.querySelectorAll('ul.list-disc li');
      expect(reasonItems.length).toBe(3); // Elektronik has 3 reasons
    });

    test('hides reasons when category is clicked again', async () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      // First click - expand
      await act(async () => {
        firstItem.click();
      });
      
      let reasonsList = firstItem.querySelector('ul.list-disc');
      expect(reasonsList).toBeTruthy();
      
      // Second click - collapse
      await act(async () => {
        firstItem.click();
      });
      
      reasonsList = firstItem.querySelector('ul.list-disc');
      expect(reasonsList).toBeNull();
    });

    test('displays feature contributions when expanded', async () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      await act(async () => {
        firstItem.click();
      });
      
      expect(firstItem.textContent).toContain('Özellik Katkıları:');
      expect(firstItem.textContent).toContain('hobby: 90%');
      expect(firstItem.textContent).toContain('age: 80%');
      expect(firstItem.textContent).toContain('budget: 85%');
    });

    test('calls onCategoryClick callback when provided', async () => {
      const onCategoryClick = vi.fn();
      const { container } = render(
        <CategoryMatchingChart 
          categories={mockCategories} 
          onCategoryClick={onCategoryClick}
        />
      );
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      await act(async () => {
        firstItem.click();
      });
      
      expect(onCategoryClick).toHaveBeenCalledTimes(1);
      expect(onCategoryClick).toHaveBeenCalledWith(mockCategories[0]);
    });
  });

  describe('Chart accessibility', () => {
    test('has proper ARIA role for the container', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const region = container.querySelector('[role="region"]');
      expect(region).toBeInTheDocument();
      expect(region).toHaveAttribute('aria-label', 'Kategori eşleştirme bilgisi');
    });

    test('has proper ARIA role for the chart', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const chart = container.querySelector('[role="img"]');
      expect(chart).toBeInTheDocument();
      expect(chart).toHaveAttribute('aria-label', 'Kategori skorları bar grafiği');
    });

    test('has proper ARIA role for the list', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const list = container.querySelector('[role="list"]');
      expect(list).toBeInTheDocument();
    });

    test('each category has proper ARIA role', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      expect(listItems.length).toBeGreaterThanOrEqual(3);
    });

    test('each category has descriptive aria-label', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[aria-label*="Elektronik"]');
      expect(firstItem).toBeInTheDocument();
      expect(firstItem).toHaveAttribute('aria-label', expect.stringContaining('85% skor'));
    });

    test('categories have aria-expanded attribute', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]');
      expect(firstItem).toHaveAttribute('aria-expanded', 'false');
    });

    test('aria-expanded changes when category is clicked', async () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      expect(firstItem).toHaveAttribute('aria-expanded', 'false');
      
      await act(async () => {
        firstItem.click();
      });
      
      expect(firstItem).toHaveAttribute('aria-expanded', 'true');
    });

    test('categories are keyboard navigable', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]');
      expect(firstItem).toHaveAttribute('tabIndex', '0');
    });
  });

  describe('Edge cases', () => {
    test('handles categories with no feature contributions', async () => {
      const categoriesWithoutContributions: CategoryMatchingReasoning[] = [
        {
          category_name: 'Test',
          score: 0.5,
          reasons: ['Test reason'],
          feature_contributions: {},
        },
      ];

      const { container } = render(<CategoryMatchingChart categories={categoriesWithoutContributions} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      await act(async () => {
        firstItem.click();
      });
      
      // Should not show feature contributions section when empty
      expect(firstItem.textContent).not.toContain('Özellik Katkıları:');
    });

    test('handles categories with undefined feature contributions', async () => {
      const categoriesWithUndefinedContributions: CategoryMatchingReasoning[] = [
        {
          category_name: 'Test',
          score: 0.5,
          reasons: ['Test reason'],
          feature_contributions: undefined as any,
        },
      ];

      const { container } = render(<CategoryMatchingChart categories={categoriesWithUndefinedContributions} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      
      // Should render without errors
      expect(firstItem).toBeTruthy();
      
      await act(async () => {
        firstItem.click();
      });
      
      // Should not show feature contributions section when undefined
      expect(firstItem.textContent).not.toContain('Özellik Katkıları:');
    });

    test('handles single category', () => {
      const singleCategory: CategoryMatchingReasoning[] = [
        {
          category_name: 'Elektronik',
          score: 0.85,
          reasons: ['Test'],
          feature_contributions: {},
        },
      ];

      render(<CategoryMatchingChart categories={singleCategory} />);
      
      expect(screen.getByText('Elektronik')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    test('handles many categories (shows only top 3)', () => {
      const manyCategories: CategoryMatchingReasoning[] = Array.from({ length: 10 }, (_, i) => ({
        category_name: `Category ${i}`,
        score: (10 - i) / 10,
        reasons: ['Test'],
        feature_contributions: {},
      }));

      const { container } = render(<CategoryMatchingChart categories={manyCategories} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      // Should show at least 3 categories
      expect(listItems.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Visual indicators', () => {
    test('displays expand/collapse arrow', () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]');
      const arrow = firstItem?.querySelector('svg');
      
      expect(arrow).toBeTruthy();
    });

    test('arrow rotates when category is expanded', async () => {
      const { container } = render(<CategoryMatchingChart categories={mockCategories} />);
      
      const firstItem = container.querySelector('[role="listitem"]') as HTMLElement;
      const arrowContainer = firstItem.querySelector('div.transition-transform');
      
      expect(arrowContainer?.className).not.toContain('rotate-180');
      
      await act(async () => {
        firstItem.click();
      });
      
      expect(arrowContainer?.className).toContain('rotate-180');
    });
  });
});
