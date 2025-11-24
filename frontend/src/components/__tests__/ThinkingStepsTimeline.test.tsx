import { describe, test, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThinkingStepsTimeline } from '../ThinkingStepsTimeline';
import { ThinkingStep } from '@/types/reasoning';

/**
 * Unit tests for ThinkingStepsTimeline component
 */

describe('ThinkingStepsTimeline', () => {
  const mockSteps: ThinkingStep[] = [
    {
      step: 1,
      action: 'Kullanıcı profilini analiz et',
      result: 'Hobiler ve ilgi alanları belirlendi',
      insight: 'Kullanıcı açık hava aktivitelerini tercih ediyor',
    },
    {
      step: 2,
      action: 'Kategori filtreleme uygula',
      result: '5 uygun kategori seçildi',
      insight: 'Spor kategorileri kullanıcı profiliyle eşleşiyor',
    },
    {
      step: 3,
      action: 'Bütçe optimizasyonu yap',
      result: 'Fiyat aralığı belirlendi',
      insight: 'Kullanıcının bütçesi orta segment ürünlere uygun',
    },
  ];

  describe('Rendering with different step counts', () => {
    test('renders timeline with title', () => {
      render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      expect(screen.getByText('Düşünme Adımları')).toBeInTheDocument();
    });

    test('renders all steps in the timeline', () => {
      render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      expect(screen.getByText('Kullanıcı profilini analiz et')).toBeInTheDocument();
      expect(screen.getByText('Kategori filtreleme uygula')).toBeInTheDocument();
      expect(screen.getByText('Bütçe optimizasyonu yap')).toBeInTheDocument();
    });

    test('displays step numbers for all steps', () => {
      render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      expect(screen.getByText('Adım 1')).toBeInTheDocument();
      expect(screen.getByText('Adım 2')).toBeInTheDocument();
      expect(screen.getByText('Adım 3')).toBeInTheDocument();
    });

    test('displays results for all steps', () => {
      render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      expect(screen.getByText(/Hobiler ve ilgi alanları belirlendi/)).toBeInTheDocument();
      expect(screen.getByText(/5 uygun kategori seçildi/)).toBeInTheDocument();
      expect(screen.getByText(/Fiyat aralığı belirlendi/)).toBeInTheDocument();
    });

    test('renders single step correctly', () => {
      const singleStep: ThinkingStep[] = [
        {
          step: 1,
          action: 'Tek adım',
          result: 'Tek sonuç',
          insight: 'İçgörü',
        },
      ];

      render(<ThinkingStepsTimeline steps={singleStep} />);
      
      expect(screen.getByText('Tek adım')).toBeInTheDocument();
      expect(screen.getByText(/Tek sonuç/)).toBeInTheDocument();
    });

    test('renders many steps with scrollable container', () => {
      const manySteps: ThinkingStep[] = Array.from({ length: 10 }, (_, i) => ({
        step: i + 1,
        action: `Adım ${i + 1} aksiyonu`,
        result: `Adım ${i + 1} sonucu`,
        insight: `Adım ${i + 1} içgörüsü`,
      }));

      const { container } = render(<ThinkingStepsTimeline steps={manySteps} />);
      
      const scrollableContainer = container.querySelector('.overflow-y-auto');
      expect(scrollableContainer).toBeInTheDocument();
      expect(scrollableContainer).toHaveClass('max-h-[600px]');
    });

    test('renders empty state when no steps provided', () => {
      render(<ThinkingStepsTimeline steps={[]} />);
      
      expect(screen.getByText('Düşünme adımı bilgisi mevcut değil')).toBeInTheDocument();
    });

    test('sorts steps chronologically even if provided out of order', () => {
      const outOfOrderSteps: ThinkingStep[] = [
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
      ];

      const { container } = render(<ThinkingStepsTimeline steps={outOfOrderSteps} />);
      
      const stepElements = container.querySelectorAll('[data-step]');
      const stepNumbers = Array.from(stepElements).map((el) =>
        parseInt(el.getAttribute('data-step') || '0', 10)
      );
      
      expect(stepNumbers).toEqual([1, 2, 3]);
    });
  });

  describe('Expand/collapse functionality', () => {
    test('steps are initially collapsed', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const buttons = container.querySelectorAll('[role="button"]');
      buttons.forEach((button) => {
        expect(button.getAttribute('aria-expanded')).toBe('false');
      });
    });

    test('clicking a step expands it', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      expect(firstStepButton).toBeInTheDocument();
      
      if (firstStepButton) {
        fireEvent.click(firstStepButton);
        
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('true');
      }
    });

    test('clicking an expanded step collapses it', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      expect(firstStepButton).toBeInTheDocument();
      
      if (firstStepButton) {
        // Expand
        fireEvent.click(firstStepButton);
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('true');
        
        // Collapse
        fireEvent.click(firstStepButton);
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('false');
      }
    });

    test('expanding a step shows insight', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        // Initially, insight should not be visible
        expect(screen.queryByText(/Kullanıcı açık hava aktivitelerini tercih ediyor/)).not.toBeInTheDocument();
        
        // Expand
        fireEvent.click(firstStepButton);
        
        // Now insight should be visible
        expect(screen.getByText(/Kullanıcı açık hava aktivitelerini tercih ediyor/)).toBeInTheDocument();
      }
    });

    test('only one step can be expanded at a time', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      const secondStepButton = container.querySelector('[data-step="2"] [role="button"]');
      
      if (firstStepButton && secondStepButton) {
        // Expand first step
        fireEvent.click(firstStepButton);
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('true');
        
        // Expand second step
        fireEvent.click(secondStepButton);
        expect(secondStepButton.getAttribute('aria-expanded')).toBe('true');
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('false');
      }
    });

    test('calls onStepClick callback when step is clicked', () => {
      const onStepClick = vi.fn();
      const { container } = render(
        <ThinkingStepsTimeline steps={mockSteps} onStepClick={onStepClick} />
      );
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.click(firstStepButton);
        
        expect(onStepClick).toHaveBeenCalledTimes(1);
        expect(onStepClick).toHaveBeenCalledWith(mockSteps[0]);
      }
    });
  });

  describe('Keyboard navigation', () => {
    test('steps are keyboard focusable', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const buttons = container.querySelectorAll('[role="button"]');
      buttons.forEach((button) => {
        expect(button.getAttribute('tabindex')).toBe('0');
      });
    });

    test('pressing Enter expands a step', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.keyDown(firstStepButton, { key: 'Enter' });
        
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('true');
      }
    });

    test('pressing Space expands a step', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.keyDown(firstStepButton, { key: ' ' });
        
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('true');
      }
    });

    test('pressing other keys does not expand a step', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.keyDown(firstStepButton, { key: 'a' });
        
        expect(firstStepButton.getAttribute('aria-expanded')).toBe('false');
      }
    });

    test('displays keyboard navigation hint', () => {
      render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      expect(
        screen.getByText(/Klavye ile gezinmek için Tab, Enter veya Space tuşlarını kullanın/)
      ).toBeInTheDocument();
    });

    test('does not display keyboard hint when no steps', () => {
      render(<ThinkingStepsTimeline steps={[]} />);
      
      expect(
        screen.queryByText(/Klavye ile gezinmek için Tab, Enter veya Space tuşlarını kullanın/)
      ).not.toBeInTheDocument();
    });
  });

  describe('Visual indicators', () => {
    test('all steps show green checkmark', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const checkmarks = container.querySelectorAll('path[d="M5 13l4 4L19 7"]');
      expect(checkmarks.length).toBe(mockSteps.length);
    });

    test('all steps have green marker background', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const greenMarkers = container.querySelectorAll('.bg-green-500');
      expect(greenMarkers.length).toBe(mockSteps.length);
    });

    test('all steps have green border', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const greenBorders = container.querySelectorAll('.border-green-600');
      expect(greenBorders.length).toBe(mockSteps.length);
    });

    test('expanded step has blue styling', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.click(firstStepButton);
        
        const firstStepContainer = container.querySelector('[data-step="1"]');
        const stepElement = firstStepContainer?.querySelector('[role="button"]');
        
        expect(stepElement?.className).toContain('border-blue-300');
        expect(stepElement?.className).toContain('bg-blue-50');
      }
    });

    test('timeline line is present between steps', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      // Timeline lines should be present for all steps except the last one
      const timelineLines = container.querySelectorAll('.bg-gray-200');
      expect(timelineLines.length).toBeGreaterThan(0);
    });

    test('expand/collapse indicator rotates when expanded', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        // Find the indicator div (the one with transition-transform class)
        const indicator = firstStepButton.querySelector('.transition-transform');
        
        // Initially not rotated
        expect(indicator?.className).not.toContain('rotate-180');
        
        // Expand
        fireEvent.click(firstStepButton);
        
        // Should be rotated
        expect(indicator?.className).toContain('rotate-180');
      }
    });
  });

  describe('Accessibility attributes', () => {
    test('has proper ARIA role for the container', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const region = container.querySelector('[role="region"]');
      expect(region).toBeInTheDocument();
      expect(region).toHaveAttribute('aria-label', 'Düşünme adımları zaman çizelgesi');
    });

    test('has proper ARIA role for the list', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const list = container.querySelector('[role="list"]');
      expect(list).toBeInTheDocument();
    });

    test('each step has proper ARIA role', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const listItems = container.querySelectorAll('[role="listitem"]');
      expect(listItems.length).toBe(mockSteps.length);
    });

    test('each step has button role for interaction', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const buttons = container.querySelectorAll('[role="button"]');
      expect(buttons.length).toBe(mockSteps.length);
    });

    test('each button has aria-expanded attribute', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const buttons = container.querySelectorAll('[role="button"]');
      buttons.forEach((button) => {
        expect(button.getAttribute('aria-expanded')).toBeTruthy();
      });
    });

    test('each button has descriptive aria-label', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstButton = container.querySelector('[data-step="1"] [role="button"]');
      expect(firstButton).toHaveAttribute(
        'aria-label',
        'Adım 1: Kullanıcı profilini analiz et'
      );
    });

    test('expanded details have region role', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      const firstStepButton = container.querySelector('[data-step="1"] [role="button"]');
      
      if (firstStepButton) {
        fireEvent.click(firstStepButton);
        
        const detailsRegion = container.querySelector('[aria-label="Adım detayları"]');
        expect(detailsRegion).toBeInTheDocument();
        expect(detailsRegion).toHaveAttribute('role', 'region');
      }
    });
  });

  describe('Custom styling', () => {
    test('applies custom className', () => {
      const { container } = render(
        <ThinkingStepsTimeline steps={mockSteps} className="custom-class" />
      );
      
      const mainContainer = container.querySelector('.custom-class');
      expect(mainContainer).toBeInTheDocument();
    });
  });

  describe('Data attributes', () => {
    test('each step has data-step attribute', () => {
      const { container } = render(<ThinkingStepsTimeline steps={mockSteps} />);
      
      mockSteps.forEach((step) => {
        const stepElement = container.querySelector(`[data-step="${step.step}"]`);
        expect(stepElement).toBeInTheDocument();
      });
    });
  });
});
