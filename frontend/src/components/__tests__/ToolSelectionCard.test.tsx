import { describe, test, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ToolSelectionCard } from '../ToolSelectionCard';
import { ToolSelectionReasoning } from '@/types/reasoning';

/**
 * Unit tests for ToolSelectionCard component
 */

describe('ToolSelectionCard', () => {
  const mockToolSelection: ToolSelectionReasoning[] = [
    {
      name: 'review_analysis',
      selected: true,
      score: 0.85,
      reason: 'Yüksek rating eşleşmesi',
      confidence: 0.9,
      priority: 1,
      factors: {
        rating_match: 0.9,
        review_sentiment: 0.85,
      },
    },
    {
      name: 'trend_analysis',
      selected: false,
      score: 0.45,
      reason: 'Düşük trend relevansı',
      confidence: 0.4,
      priority: 2,
      factors: {
        trend_score: 0.5,
      },
    },
    {
      name: 'inventory_check',
      selected: true,
      score: 0.75,
      reason: 'Stok mevcut',
      confidence: 0.8,
      priority: 3,
    },
  ];

  describe('Rendering with different tool states', () => {
    test('renders tool selection card with title', () => {
      render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      expect(screen.getByText('Tool Seçimi')).toBeInTheDocument();
    });

    test('renders all tools in the list', () => {
      render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      expect(screen.getByText('Yorum Analizi')).toBeInTheDocument();
      expect(screen.getByText('Trend Analizi')).toBeInTheDocument();
      expect(screen.getByText('Stok Kontrolü')).toBeInTheDocument();
    });

    test('displays confidence scores for all tools', () => {
      render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      expect(screen.getByText('90%')).toBeInTheDocument();
      expect(screen.getByText('40%')).toBeInTheDocument();
      expect(screen.getByText('80%')).toBeInTheDocument();
    });

    test('displays priority for all tools', () => {
      render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      expect(screen.getByText('Öncelik: 1')).toBeInTheDocument();
      expect(screen.getByText('Öncelik: 2')).toBeInTheDocument();
      expect(screen.getByText('Öncelik: 3')).toBeInTheDocument();
    });

    test('renders selected tools with green styling', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const selectedItems = Array.from(listItems).filter((item) =>
        item.className.includes('border-green-300')
      );
      
      expect(selectedItems.length).toBe(2); // review_analysis and inventory_check
    });

    test('renders unselected tools with gray styling', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const unselectedItems = Array.from(listItems).filter((item) =>
        item.className.includes('border-gray-200')
      );
      
      expect(unselectedItems.length).toBe(1); // trend_analysis
    });

    test('shows low confidence warning for tools with confidence < 0.5', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const warningIcons = container.querySelectorAll('.bg-yellow-100');
      expect(warningIcons.length).toBe(1); // Only trend_analysis has confidence 0.4
    });

    test('does not show low confidence warning for tools with confidence >= 0.5', () => {
      const toolsWithHighConfidence: ToolSelectionReasoning[] = [
        {
          name: 'review_analysis',
          selected: true,
          score: 0.85,
          reason: 'Good match',
          confidence: 0.9,
          priority: 1,
        },
        {
          name: 'trend_analysis',
          selected: true,
          score: 0.75,
          reason: 'Trending',
          confidence: 0.8,
          priority: 2,
        },
      ];

      const { container } = render(<ToolSelectionCard toolSelection={toolsWithHighConfidence} />);
      
      const warningIcons = container.querySelectorAll('.bg-yellow-100');
      expect(warningIcons.length).toBe(0);
    });

    test('renders empty state when no tools provided', () => {
      render(<ToolSelectionCard toolSelection={[]} />);
      
      expect(screen.getByText('Tool seçim bilgisi mevcut değil')).toBeInTheDocument();
    });

    test('sorts tools by priority', () => {
      const unsortedTools: ToolSelectionReasoning[] = [
        {
          name: 'trend_analysis',
          selected: false,
          score: 0.5,
          reason: 'Test',
          confidence: 0.6,
          priority: 3,
        },
        {
          name: 'review_analysis',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 1,
        },
        {
          name: 'inventory_check',
          selected: true,
          score: 0.7,
          reason: 'Test',
          confidence: 0.8,
          priority: 2,
        },
      ];

      const { container } = render(<ToolSelectionCard toolSelection={unsortedTools} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      const toolNames = Array.from(listItems).map((item) => item.textContent);
      
      // First item should be review_analysis (priority 1)
      expect(toolNames[0]).toContain('Yorum Analizi');
      // Second item should be inventory_check (priority 2)
      expect(toolNames[1]).toContain('Stok Kontrolü');
      // Third item should be trend_analysis (priority 3)
      expect(toolNames[2]).toContain('Trend Analizi');
    });
  });

  describe('Accessibility attributes', () => {
    test('has proper ARIA role for the container', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const region = container.querySelector('[role="region"]');
      expect(region).toBeInTheDocument();
      expect(region).toHaveAttribute('aria-label', 'Tool seçim bilgisi');
    });

    test('has proper ARIA role for the list', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const list = container.querySelector('[role="list"]');
      expect(list).toBeInTheDocument();
    });

    test('each tool has proper ARIA role', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      expect(listItems.length).toBe(3);
    });

    test('each tool has descriptive aria-label', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const firstTool = container.querySelector('[aria-label*="Yorum Analizi"]');
      expect(firstTool).toBeInTheDocument();
      expect(firstTool).toHaveAttribute('aria-label', expect.stringContaining('seçildi'));
      expect(firstTool).toHaveAttribute('aria-label', expect.stringContaining('90%'));
    });

    test('low confidence warning has aria-label', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const warningIcon = container.querySelector('[aria-label="Düşük güven uyarısı"]');
      expect(warningIcon).toBeInTheDocument();
    });
  });

  describe('Tool display names', () => {
    test('translates tool names to Turkish', () => {
      const allTools: ToolSelectionReasoning[] = [
        {
          name: 'review_analysis',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 1,
        },
        {
          name: 'trend_analysis',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 2,
        },
        {
          name: 'inventory_check',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 3,
        },
        {
          name: 'price_comparison',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 4,
        },
        {
          name: 'category_filter',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 5,
        },
      ];

      render(<ToolSelectionCard toolSelection={allTools} />);
      
      expect(screen.getByText('Yorum Analizi')).toBeInTheDocument();
      expect(screen.getByText('Trend Analizi')).toBeInTheDocument();
      expect(screen.getByText('Stok Kontrolü')).toBeInTheDocument();
      expect(screen.getByText('Fiyat Karşılaştırma')).toBeInTheDocument();
      expect(screen.getByText('Kategori Filtresi')).toBeInTheDocument();
    });

    test('displays unknown tool names as-is', () => {
      const unknownTool: ToolSelectionReasoning[] = [
        {
          name: 'unknown_tool',
          selected: true,
          score: 0.8,
          reason: 'Test',
          confidence: 0.9,
          priority: 1,
        },
      ];

      render(<ToolSelectionCard toolSelection={unknownTool} />);
      
      expect(screen.getByText('unknown_tool')).toBeInTheDocument();
    });
  });

  describe('Tool factors', () => {
    test('handles tools without factors', () => {
      const toolWithoutFactors: ToolSelectionReasoning[] = [
        {
          name: 'review_analysis',
          selected: true,
          score: 0.8,
          reason: 'Test reason',
          confidence: 0.9,
          priority: 1,
        },
      ];

      const { container } = render(<ToolSelectionCard toolSelection={toolWithoutFactors} />);
      
      // Should render without errors
      expect(container.querySelector('[role="listitem"]')).toBeInTheDocument();
    });

    test('handles tools with empty factors object', () => {
      const toolWithEmptyFactors: ToolSelectionReasoning[] = [
        {
          name: 'review_analysis',
          selected: true,
          score: 0.8,
          reason: 'Test reason',
          confidence: 0.9,
          priority: 1,
          factors: {},
        },
      ];

      const { container } = render(<ToolSelectionCard toolSelection={toolWithEmptyFactors} />);
      
      // Should render without errors
      expect(container.querySelector('[role="listitem"]')).toBeInTheDocument();
    });
  });

  describe('Visual indicators', () => {
    test('selected tools show checkmark icon', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      // Check for checkmark SVG path
      const checkmarks = container.querySelectorAll('path[d="M5 13l4 4L19 7"]');
      expect(checkmarks.length).toBe(2); // review_analysis and inventory_check are selected
    });

    test('unselected tools show cross icon', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      // Check for cross SVG path
      const crosses = container.querySelectorAll('path[d="M6 18L18 6M6 6l12 12"]');
      expect(crosses.length).toBe(1); // Only trend_analysis is unselected
    });

    test('selected tools have green icon background', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const greenIcons = container.querySelectorAll('.bg-green-500');
      expect(greenIcons.length).toBe(2); // review_analysis and inventory_check
    });

    test('unselected tools have gray icon background', () => {
      const { container } = render(<ToolSelectionCard toolSelection={mockToolSelection} />);
      
      const grayIcons = container.querySelectorAll('.bg-gray-300');
      expect(grayIcons.length).toBe(1); // trend_analysis
    });
  });
});
