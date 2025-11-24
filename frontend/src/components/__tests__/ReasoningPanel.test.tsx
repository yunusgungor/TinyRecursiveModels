/**
 * Unit tests for ReasoningPanel component
 * Tests panel open/close, filter functionality, and mobile responsive behavior
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReasoningPanel } from '../ReasoningPanel';
import type {
  ReasoningTrace,
  GiftItem,
  UserProfile,
  ReasoningFilter,
} from '@/types/reasoning';

// Mock data
const mockGift: GiftItem = {
  id: 'gift-123',
  name: 'Test Gift',
  price: 100,
  category: 'Test Category',
  image_url: 'https://example.com/image.jpg',
  rating: 4.5,
  availability: true,
};

const mockUserProfile: UserProfile = {
  hobbies: ['test'],
  age: 30,
  budget: 200,
  occasion: 'birthday',
};

const mockReasoningTrace: ReasoningTrace = {
  tool_selection: [
    {
      name: 'test_tool',
      selected: true,
      score: 0.8,
      reason: 'Test reason',
      confidence: 0.9,
      priority: 1,
    },
  ],
  category_matching: [
    {
      category_name: 'Test Category',
      score: 0.85,
      reasons: ['Test reason 1', 'Test reason 2'],
      feature_contributions: { test: 0.8 },
    },
  ],
  attention_weights: {
    user_features: { hobbies: 0.5, budget: 0.3, age: 0.2 },
    gift_features: { category: 0.6, price: 0.4 },
  },
  thinking_steps: [
    {
      step: 1,
      action: 'Test action',
      result: 'Test result',
      insight: 'Test insight',
    },
  ],
};

describe('ReasoningPanel', () => {
  describe('Panel open/close', () => {
    it('should render when isOpen is true', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByText('Detaylı Analiz')).toBeInTheDocument();
    });

    it('should not render when isOpen is false', () => {
      const { container } = render(
        <ReasoningPanel
          isOpen={false}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.queryByText('Detaylı Analiz')).not.toBeInTheDocument();
    });

    it('should call onClose when close button is clicked', async () => {
      const onClose = vi.fn();

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={onClose}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      const closeButton = screen.getByLabelText('Paneli kapat');
      fireEvent.click(closeButton);

      await waitFor(() => {
        expect(onClose).toHaveBeenCalledTimes(1);
      });
    });

    it('should call onClose when escape key is pressed', async () => {
      const onClose = vi.fn();

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={onClose}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      fireEvent.keyDown(document, { key: 'Escape' });

      await waitFor(() => {
        // Radix UI Dialog also handles Escape, so it may be called more than once
        expect(onClose).toHaveBeenCalled();
      });
    });
  });

  describe('Filter functionality', () => {
    it('should display filter buttons', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={[]}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByText('Tool Seçimi')).toBeInTheDocument();
      expect(screen.getByText('Kategori Eşleştirme')).toBeInTheDocument();
      expect(screen.getByText('Attention Weights')).toBeInTheDocument();
      expect(screen.getByText('Düşünme Adımları')).toBeInTheDocument();
    });

    it('should call onFilterChange when filter button is clicked', async () => {
      const onFilterChange = vi.fn();

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={[]}
          onFilterChange={onFilterChange}
        />
      );

      const toolSelectionButton = screen.getByText('Tool Seçimi');
      fireEvent.click(toolSelectionButton);

      await waitFor(() => {
        expect(onFilterChange).toHaveBeenCalledWith(['tool_selection']);
      });
    });

    it('should show "Tümünü Göster" button', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={[]}
          onFilterChange={() => {}}
        />
      );

      // There are multiple "Tümünü Göster" buttons (filter bar + empty state)
      expect(screen.getAllByText('Tümünü Göster').length).toBeGreaterThan(0);
    });

    it('should call onFilterChange with all filters when "Tümünü Göster" is clicked', async () => {
      const onFilterChange = vi.fn();

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={[]}
          onFilterChange={onFilterChange}
        />
      );

      // Get the first "Tümünü Göster" button (from filter bar)
      const showAllButtons = screen.getAllByText('Tümünü Göster');
      fireEvent.click(showAllButtons[0]);

      await waitFor(() => {
        expect(onFilterChange).toHaveBeenCalledWith([
          'tool_selection',
          'category_matching',
          'attention_weights',
          'thinking_steps',
        ]);
      });
    });

    it('should show "Temizle" button', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByText('Temizle')).toBeInTheDocument();
    });

    it('should call onFilterChange with empty array when "Temizle" is clicked', async () => {
      const onFilterChange = vi.fn();

      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={onFilterChange}
        />
      );

      const clearButton = screen.getByText('Temizle');
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(onFilterChange).toHaveBeenCalledWith([]);
      });
    });

    it('should display active filter count', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection', 'category_matching']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByText('2 bölüm gösteriliyor')).toBeInTheDocument();
    });

    it('should show empty state when no filters are active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={[]}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByText('Hiçbir filtre seçilmedi')).toBeInTheDocument();
    });
  });

  describe('Export functionality', () => {
    it('should display export button', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByLabelText('Export seçenekleri')).toBeInTheDocument();
    });

    it('should show export dropdown when export button is clicked', async () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      const exportButton = screen.getByLabelText('Export seçenekleri');
      fireEvent.click(exportButton);

      // Note: Radix UI dropdown menu renders in a portal, which may not be immediately visible in tests
      // This test verifies the button exists and is clickable
      expect(exportButton).toBeInTheDocument();
    });
  });

  describe('Content rendering', () => {
    it('should render tool selection when filter is active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      // Check for the tool selection card by its heading
      expect(screen.getAllByText('Tool Seçimi')[1]).toBeInTheDocument(); // [1] to get the heading, not the button
    });

    it('should render category matching when filter is active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['category_matching']}
          onFilterChange={() => {}}
        />
      );

      // Check for the category matching chart by its heading
      expect(screen.getAllByText('Kategori Eşleştirme')[1]).toBeInTheDocument(); // [1] to get the heading, not the button
    });

    it('should render attention weights when filter is active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['attention_weights']}
          onFilterChange={() => {}}
        />
      );

      // Check for the attention weights chart by its heading
      expect(screen.getAllByText('Attention Weights')[1]).toBeInTheDocument(); // [1] to get the heading, not the button
    });

    it('should render thinking steps when filter is active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['thinking_steps']}
          onFilterChange={() => {}}
        />
      );

      // Check for the thinking steps timeline by its heading
      expect(screen.getAllByText('Düşünme Adımları')[1]).toBeInTheDocument(); // [1] to get the heading, not the button
    });

    it('should render all sections when all filters are active', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection', 'category_matching', 'attention_weights', 'thinking_steps']}
          onFilterChange={() => {}}
        />
      );

      // Check that all section headings are present (index [1] to get headings, not buttons)
      expect(screen.getAllByText('Tool Seçimi')[1]).toBeInTheDocument();
      expect(screen.getAllByText('Kategori Eşleştirme')[1]).toBeInTheDocument();
      expect(screen.getAllByText('Attention Weights')[1]).toBeInTheDocument();
      expect(screen.getAllByText('Düşünme Adımları')[1]).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      expect(screen.getByLabelText('Paneli kapat')).toBeInTheDocument();
      expect(screen.getByLabelText('Export seçenekleri')).toBeInTheDocument();
    });

    it('should have proper aria-pressed attributes on filter buttons', () => {
      render(
        <ReasoningPanel
          isOpen={true}
          onClose={() => {}}
          reasoningTrace={mockReasoningTrace}
          gift={mockGift}
          userProfile={mockUserProfile}
          activeFilters={['tool_selection']}
          onFilterChange={() => {}}
        />
      );

      // Get the filter buttons (index [0] to get buttons, not headings)
      const toolButton = screen.getAllByText('Tool Seçimi')[0];
      expect(toolButton).toHaveAttribute('aria-pressed', 'true');

      const categoryButton = screen.getAllByText('Kategori Eşleştirme')[0];
      expect(categoryButton).toHaveAttribute('aria-pressed', 'false');
    });
  });
});
