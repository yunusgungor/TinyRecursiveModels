import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeToggle } from '../ThemeToggle';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { useAppStore } from '@/store/useAppStore';

describe('ThemeToggle', () => {
  beforeEach(() => {
    localStorage.clear();
    useAppStore.setState({ theme: 'light' });
  });

  afterEach(() => {
    document.documentElement.classList.remove('dark');
  });

  const renderWithTheme = () => {
    return render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>
    );
  };

  it('should render theme toggle button', () => {
    renderWithTheme();
    
    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
  });

  it('should show moon icon when theme is light', () => {
    renderWithTheme();
    
    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-label', 'Karanlık moda geç');
    expect(button).toHaveAttribute('title', 'Karanlık moda geç');
  });

  it('should show sun icon when theme is dark', () => {
    useAppStore.setState({ theme: 'dark' });
    
    renderWithTheme();
    
    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-label', 'Aydınlık moda geç');
    expect(button).toHaveAttribute('title', 'Aydınlık moda geç');
  });

  it('should toggle theme when clicked', async () => {
    const user = userEvent.setup();
    renderWithTheme();
    
    const button = screen.getByRole('button');
    
    // Initially light
    expect(useAppStore.getState().theme).toBe('light');
    
    // Click to toggle to dark
    await user.click(button);
    expect(useAppStore.getState().theme).toBe('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    
    // Click to toggle back to light
    await user.click(button);
    expect(useAppStore.getState().theme).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('should update icon when theme changes', async () => {
    const user = userEvent.setup();
    renderWithTheme();
    
    const button = screen.getByRole('button');
    
    // Initially shows moon icon (light theme)
    expect(button).toHaveAttribute('aria-label', 'Karanlık moda geç');
    
    // Click to toggle
    await user.click(button);
    
    // Now shows sun icon (dark theme)
    expect(button).toHaveAttribute('aria-label', 'Aydınlık moda geç');
  });

  it('should have proper styling classes', () => {
    renderWithTheme();
    
    const button = screen.getByRole('button');
    
    // Check for fixed positioning
    expect(button).toHaveClass('fixed');
    expect(button).toHaveClass('top-4');
    expect(button).toHaveClass('right-4');
    
    // Check for z-index
    expect(button).toHaveClass('z-50');
    
    // Check for styling
    expect(button).toHaveClass('rounded-full');
    expect(button).toHaveClass('shadow-lg');
  });

  it('should be keyboard accessible', async () => {
    const user = userEvent.setup();
    renderWithTheme();
    
    const button = screen.getByRole('button');
    
    // Focus the button
    button.focus();
    expect(button).toHaveFocus();
    
    // Press Enter to toggle
    await user.keyboard('{Enter}');
    expect(useAppStore.getState().theme).toBe('dark');
    
    // Press Space to toggle
    await user.keyboard(' ');
    expect(useAppStore.getState().theme).toBe('light');
  });

  it('should have focus styles', () => {
    renderWithTheme();
    
    const button = screen.getByRole('button');
    
    // Check for focus ring classes
    expect(button).toHaveClass('focus:outline-none');
    expect(button).toHaveClass('focus:ring-2');
    expect(button).toHaveClass('focus:ring-blue-500');
  });
});
