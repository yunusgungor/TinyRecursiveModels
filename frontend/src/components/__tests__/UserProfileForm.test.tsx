import { describe, test, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UserProfileForm } from '../UserProfileForm';

describe('UserProfileForm', () => {
  describe('Component Rendering', () => {
    test('renders all form fields', () => {
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      expect(screen.getByLabelText(/yaş/i)).toBeInTheDocument();
      expect(screen.getByText(/hobiler/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/lişki durumu/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/bütçe/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/özel gün/i)).toBeInTheDocument();
      expect(screen.getByText(/kişilik özellikleri/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /hediye önerisi al/i })).toBeInTheDocument();
    });

    test('renders hobby options', () => {
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      expect(screen.getByRole('button', { name: 'Spor' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Müzik' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Okuma' })).toBeInTheDocument();
    });

    test('disables form when loading', () => {
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} isLoading={true} />);

      expect(screen.getByLabelText(/yaş/i)).toBeDisabled();
      expect(screen.getByRole('button', { name: /yükleniyor/i })).toBeDisabled();
    });
  });

  describe('User Interactions', () => {
    test('allows age input', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const ageInput = screen.getByLabelText(/yaş/i);
      await user.type(ageInput, '25');

      expect(ageInput).toHaveValue(25);
    });

    test('allows hobby selection', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const sporButton = screen.getByRole('button', { name: 'Spor' });
      await user.click(sporButton);

      expect(sporButton).toHaveClass('bg-blue-600');
    });

    test('allows hobby deselection', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const sporButton = screen.getByRole('button', { name: 'Spor' });
      await user.click(sporButton);
      expect(sporButton).toHaveClass('bg-blue-600');

      await user.click(sporButton);
      expect(sporButton).not.toHaveClass('bg-blue-600');
    });

    test('allows relationship selection', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const relationshipSelect = screen.getByLabelText(/lişki durumu/i);
      await user.selectOptions(relationshipSelect, 'Anne');

      expect(relationshipSelect).toHaveValue('Anne');
    });

    test('allows budget input', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const budgetInput = screen.getByLabelText(/bütçe/i);
      await user.type(budgetInput, '500');

      expect(budgetInput).toHaveValue('500');
    });

    test('formats budget on blur', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const budgetInput = screen.getByLabelText(/bütçe/i);
      await user.type(budgetInput, '500');
      await user.tab();

      await waitFor(() => {
        const value = budgetInput.value;
        expect(value).toContain('₺');
      });
    });

    test('allows occasion selection', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const occasionSelect = screen.getByLabelText(/özel gün/i);
      await user.selectOptions(occasionSelect, 'Doğum Günü');

      expect(occasionSelect).toHaveValue('Doğum Günü');
    });

    test('allows personality trait selection', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const pratikButton = screen.getByRole('button', { name: 'Pratik' });
      await user.click(pratikButton);

      expect(pratikButton).toHaveClass('bg-green-600');
    });

    test('limits personality traits to 5', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      // Select 5 traits
      await user.click(screen.getByRole('button', { name: 'Pratik' }));
      await user.click(screen.getByRole('button', { name: 'Romantik' }));
      await user.click(screen.getByRole('button', { name: 'Sportif' }));
      await user.click(screen.getByRole('button', { name: 'Entelektüel' }));
      await user.click(screen.getByRole('button', { name: 'Sanatsal' }));

      // Try to select 6th trait
      const sixthButton = screen.getByRole('button', { name: 'Teknoloji Meraklısı' });
      expect(sixthButton).toHaveClass('opacity-50');
    });
  });

  describe('Validation', () => {
    test('shows error for invalid age', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const ageInput = screen.getByLabelText(/yaş/i);
      await user.type(ageInput, '10');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/yaş en az 18 olmalıdır/i)).toBeInTheDocument();
      });
    });

    test('shows error for age over 100', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const ageInput = screen.getByLabelText(/yaş/i);
      await user.type(ageInput, '150');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/yaş en fazla 100 olabilir/i)).toBeInTheDocument();
      });
    });

    test('shows error for negative budget', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const budgetInput = screen.getByLabelText(/bütçe/i);
      await user.type(budgetInput, '-100');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/bütçe pozitif bir değer olmalıdır/i)).toBeInTheDocument();
      });
    });

    test('prevents submission with empty required fields', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      const submitButton = screen.getByRole('button', { name: /hediye önerisi al/i });
      await user.click(submitButton);

      expect(mockSubmit).not.toHaveBeenCalled();
    });

    test('submits form with valid data', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      // Fill all required fields
      await user.type(screen.getByLabelText(/yaş/i), '25');
      await user.click(screen.getByRole('button', { name: 'Spor' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Arkadaş');
      await user.type(screen.getByLabelText(/bütçe/i), '500');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Doğum Günü');

      const submitButton = screen.getByRole('button', { name: /hediye önerisi al/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockSubmit).toHaveBeenCalledWith({
          age: 25,
          hobbies: ['Spor'],
          relationship: 'Arkadaş',
          budget: 500,
          occasion: 'Doğum Günü',
          personalityTraits: [],
        });
      });
    });

    test('submits form with personality traits', async () => {
      const user = userEvent.setup();
      const mockSubmit = vi.fn();
      render(<UserProfileForm onSubmit={mockSubmit} />);

      // Fill all required fields
      await user.type(screen.getByLabelText(/yaş/i), '30');
      await user.click(screen.getByRole('button', { name: 'Müzik' }));
      await user.selectOptions(screen.getByLabelText(/lişki durumu/i), 'Eş');
      await user.type(screen.getByLabelText(/bütçe/i), '1000');
      await user.selectOptions(screen.getByLabelText(/özel gün/i), 'Yıldönümü');
      await user.click(screen.getByRole('button', { name: 'Romantik' }));

      const submitButton = screen.getByRole('button', { name: /hediye önerisi al/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockSubmit).toHaveBeenCalledWith({
          age: 30,
          hobbies: ['Müzik'],
          relationship: 'Eş',
          budget: 1000,
          occasion: 'Yıldönümü',
          personalityTraits: ['Romantik'],
        });
      });
    });
  });
});
