import { describe, test, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfidenceIndicator } from '../ConfidenceIndicator';

/**
 * Unit tests for ConfidenceIndicator component
 */

describe('ConfidenceIndicator', () => {
  describe('Rendering with different confidence values', () => {
    test('renders high confidence (0.9) with green styling', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.9} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator).toBeTruthy();
      
      const classList = indicator?.className || '';
      expect(classList).toContain('bg-green-100');
      expect(classList).toContain('text-green-800');
      
      expect(container.textContent).toContain('90%');
      expect(container.textContent).toContain('Yüksek Güven');
    });

    test('renders medium confidence (0.65) with yellow styling', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.65} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator).toBeTruthy();
      
      const classList = indicator?.className || '';
      expect(classList).toContain('bg-yellow-100');
      expect(classList).toContain('text-yellow-800');
      
      expect(container.textContent).toContain('65%');
      expect(container.textContent).toContain('Orta Güven');
    });

    test('renders low confidence (0.3) with red styling', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.3} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator).toBeTruthy();
      
      const classList = indicator?.className || '';
      expect(classList).toContain('bg-red-100');
      expect(classList).toContain('text-red-800');
      
      expect(container.textContent).toContain('30%');
      expect(container.textContent).toContain('Düşük Güven');
    });

    test('renders boundary value 0.8 as medium confidence', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.8} />);
      
      expect(container.textContent).toContain('80%');
      expect(container.textContent).toContain('Orta Güven');
    });

    test('renders boundary value 0.5 as medium confidence', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.5} />);
      
      expect(container.textContent).toContain('50%');
      expect(container.textContent).toContain('Orta Güven');
    });

    test('formats confidence percentage correctly', () => {
      const { container: container1 } = render(<ConfidenceIndicator confidence={0.856} />);
      expect(container1.textContent).toContain('86%');

      const { container: container2 } = render(<ConfidenceIndicator confidence={0.123} />);
      expect(container2.textContent).toContain('12%');

      const { container: container3 } = render(<ConfidenceIndicator confidence={1.0} />);
      expect(container3.textContent).toContain('100%');
    });
  });

  describe('Click handler invocation', () => {
    test('calls onClick handler when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={handleClick} />
      );
      
      const indicator = container.querySelector('[role="button"]');
      expect(indicator).toBeTruthy();
      
      await user.click(indicator!);
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('calls onClick handler when Enter key is pressed', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={handleClick} />
      );
      
      const indicator = container.querySelector('[role="button"]') as HTMLElement;
      expect(indicator).toBeTruthy();
      
      indicator.focus();
      await user.keyboard('{Enter}');
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('calls onClick handler when Space key is pressed', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={handleClick} />
      );
      
      const indicator = container.querySelector('[role="button"]') as HTMLElement;
      expect(indicator).toBeTruthy();
      
      indicator.focus();
      await user.keyboard(' ');
      
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('does not call onClick when not provided', async () => {
      const user = userEvent.setup();
      
      const { container } = render(<ConfidenceIndicator confidence={0.75} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator).toBeTruthy();
      
      // Should not throw error when clicked without onClick handler
      await user.click(indicator!);
      
      // No assertion needed - just ensuring no error is thrown
    });

    test('applies cursor-pointer class when onClick is provided', () => {
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={() => {}} />
      );
      
      const indicator = container.querySelector('[role="button"]');
      const classList = indicator?.className || '';
      
      expect(classList).toContain('cursor-pointer');
    });

    test('does not apply cursor-pointer class when onClick is not provided', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.75} />);
      
      const indicator = container.querySelector('[role="status"]');
      const classList = indicator?.className || '';
      
      expect(classList).not.toContain('cursor-pointer');
    });
  });

  describe('ARIA attributes', () => {
    test('has correct role when clickable', () => {
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={() => {}} />
      );
      
      const indicator = container.querySelector('[role="button"]');
      expect(indicator).toBeTruthy();
    });

    test('has correct role when not clickable', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.75} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator).toBeTruthy();
    });

    test('has aria-label with confidence information', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.85} />);
      
      const indicator = container.querySelector('[aria-label]');
      expect(indicator).toBeTruthy();
      
      const ariaLabel = indicator?.getAttribute('aria-label');
      expect(ariaLabel).toContain('Güven skoru');
      expect(ariaLabel).toContain('85%');
      expect(ariaLabel).toContain('Yüksek Güven');
    });

    test('has aria-live="polite" for screen readers', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.75} />);
      
      const indicator = container.querySelector('[aria-live="polite"]');
      expect(indicator).toBeTruthy();
    });

    test('is keyboard focusable when clickable', () => {
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} onClick={() => {}} />
      );
      
      const indicator = container.querySelector('[role="button"]');
      expect(indicator?.getAttribute('tabIndex')).toBe('0');
    });

    test('is not keyboard focusable when not clickable', () => {
      const { container } = render(<ConfidenceIndicator confidence={0.75} />);
      
      const indicator = container.querySelector('[role="status"]');
      expect(indicator?.hasAttribute('tabIndex')).toBe(false);
    });
  });

  describe('Custom className', () => {
    test('applies custom className', () => {
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} className="custom-class" />
      );
      
      const indicator = container.querySelector('[role="status"]');
      const classList = indicator?.className || '';
      
      expect(classList).toContain('custom-class');
    });

    test('merges custom className with default classes', () => {
      const { container } = render(
        <ConfidenceIndicator confidence={0.75} className="custom-class" />
      );
      
      const indicator = container.querySelector('[role="status"]');
      const classList = indicator?.className || '';
      
      expect(classList).toContain('custom-class');
      expect(classList).toContain('bg-yellow-100'); // Default medium confidence class
    });
  });
});
