import { describe, test, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfidenceExplanationModal } from '../ConfidenceExplanationModal';
import { ConfidenceExplanation } from '@/types';

describe('ConfidenceExplanationModal', () => {
  const mockOnClose = vi.fn();

  const highConfidenceExplanation: ConfidenceExplanation = {
    score: 0.92,
    level: 'high',
    factors: {
      positive: [
        'Perfect hobby match',
        'Within budget',
        'High rating',
      ],
      negative: [
        'Limited stock',
      ],
    },
  };

  const mediumConfidenceExplanation: ConfidenceExplanation = {
    score: 0.65,
    level: 'medium',
    factors: {
      positive: [
        'Age appropriate',
      ],
      negative: [
        'Over budget',
        'Medium rating',
      ],
    },
  };

  const lowConfidenceExplanation: ConfidenceExplanation = {
    score: 0.35,
    level: 'low',
    factors: {
      positive: [
        'In stock',
      ],
      negative: [
        'Hobby mismatch',
        'Way over budget',
        'Low rating',
      ],
    },
  };

  beforeEach(() => {
    mockOnClose.mockClear();
  });

  test('renders modal when isOpen is true', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Güven Skoru Açıklaması')).toBeInTheDocument();
  });

  test('does not render modal when isOpen is false', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={false}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('displays correct confidence score and level for high confidence', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    expect(screen.getByText('92%')).toBeInTheDocument();
    expect(screen.getByText('Yüksek Güven')).toBeInTheDocument();
  });

  test('displays correct confidence score and level for medium confidence', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={mediumConfidenceExplanation}
      />
    );

    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.getByText('Orta Güven')).toBeInTheDocument();
  });

  test('displays correct confidence score and level for low confidence', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={lowConfidenceExplanation}
      />
    );

    expect(screen.getByText('35%')).toBeInTheDocument();
    expect(screen.getByText('Düşük Güven')).toBeInTheDocument();
  });

  test('renders all positive factors', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    expect(screen.getByText('Olumlu Faktörler')).toBeInTheDocument();
    expect(screen.getByText('Perfect hobby match')).toBeInTheDocument();
    expect(screen.getByText('Within budget')).toBeInTheDocument();
    expect(screen.getByText('High rating')).toBeInTheDocument();
  });

  test('renders all negative factors', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    expect(screen.getByText('Olumsuz Faktörler')).toBeInTheDocument();
    expect(screen.getByText('Limited stock')).toBeInTheDocument();
  });

  test('calls onClose when close button is clicked', async () => {
    const user = userEvent.setup();
    
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    // Get the text button (not the X button with aria-label)
    const closeButtons = screen.getAllByRole('button', { name: 'Kapat' });
    const textButton = closeButtons.find(btn => btn.textContent === 'Kapat');
    expect(textButton).toBeDefined();
    
    await user.click(textButton!);

    // The button triggers onClose, but Radix Dialog also triggers it
    // So we just check that it was called at least once
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('calls onClose when X button is clicked', async () => {
    const user = userEvent.setup();
    
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    // Get the X button (with aria-label but no text content)
    const closeButtons = screen.getAllByRole('button', { name: 'Kapat' });
    const xButton = closeButtons.find(btn => btn.getAttribute('aria-label') === 'Kapat' && !btn.textContent);
    expect(xButton).toBeDefined();
    
    await user.click(xButton!);

    expect(mockOnClose).toHaveBeenCalled();
  });

  test('handles explanation with only positive factors', () => {
    const onlyPositive: ConfidenceExplanation = {
      score: 0.95,
      level: 'high',
      factors: {
        positive: ['Factor 1', 'Factor 2'],
        negative: [],
      },
    };

    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={onlyPositive}
      />
    );

    expect(screen.getByText('Olumlu Faktörler')).toBeInTheDocument();
    expect(screen.queryByText('Olumsuz Faktörler')).not.toBeInTheDocument();
  });

  test('handles explanation with only negative factors', () => {
    const onlyNegative: ConfidenceExplanation = {
      score: 0.15,
      level: 'low',
      factors: {
        positive: [],
        negative: ['Factor 1', 'Factor 2'],
      },
    };

    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={onlyNegative}
      />
    );

    expect(screen.queryByText('Olumlu Faktörler')).not.toBeInTheDocument();
    expect(screen.getByText('Olumsuz Faktörler')).toBeInTheDocument();
  });

  test('has proper ARIA attributes', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-describedby', 'confidence-explanation-description');
  });

  test('renders factor lists with proper roles', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
      />
    );

    const lists = screen.getAllByRole('list');
    expect(lists.length).toBeGreaterThan(0);
  });

  test('applies custom className', () => {
    render(
      <ConfidenceExplanationModal
        isOpen={true}
        onClose={mockOnClose}
        explanation={highConfidenceExplanation}
        className="custom-class"
      />
    );

    // The className is applied to the Dialog.Content which is rendered in a portal
    // We can verify the dialog exists and has the expected structure
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    
    // Check that the dialog has the custom class in the document body
    const customElement = document.body.querySelector('.custom-class');
    expect(customElement).toBeTruthy();
  });
});
