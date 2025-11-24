/**
 * Unit tests for loading and error state components
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  Spinner,
  GiftCardSkeleton,
  ToolSelectionSkeleton,
  CategoryChartSkeleton,
  AttentionWeightsSkeleton,
  ThinkingStepsSkeleton,
  ReasoningPanelSkeleton,
  LoadingOverlay,
} from '../LoadingStates';
import {
  ErrorMessage,
  InlineErrorMessage,
  ReasoningUnavailableError,
  NetworkError,
  TimeoutError,
  EmptyState,
} from '../ErrorStates';

describe('LoadingStates', () => {
  describe('Spinner', () => {
    it('should render with default size', () => {
      const { container } = render(<Spinner />);
      const spinner = container.querySelector('[role="status"]');
      expect(spinner).toBeInTheDocument();
      expect(spinner?.className).toContain('w-8');
    });

    it('should render with small size', () => {
      const { container } = render(<Spinner size="sm" />);
      const spinner = container.querySelector('[role="status"]');
      expect(spinner?.className).toContain('w-4');
    });

    it('should render with large size', () => {
      const { container } = render(<Spinner size="lg" />);
      const spinner = container.querySelector('[role="status"]');
      expect(spinner?.className).toContain('w-12');
    });

    it('should have proper accessibility attributes', () => {
      render(<Spinner />);
      const spinner = screen.getByRole('status');
      expect(spinner).toHaveAttribute('aria-label', 'Loading');
    });
  });

  describe('GiftCardSkeleton', () => {
    it('should render skeleton loader', () => {
      const { container } = render(<GiftCardSkeleton />);
      const skeleton = container.querySelector('[role="status"]');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('aria-label', 'Loading gift recommendation');
    });

    it('should have animation class', () => {
      const { container } = render(<GiftCardSkeleton />);
      const skeleton = container.querySelector('.animate-pulse');
      expect(skeleton).toBeInTheDocument();
    });
  });

  describe('ToolSelectionSkeleton', () => {
    it('should render skeleton loader', () => {
      const { container } = render(<ToolSelectionSkeleton />);
      const skeleton = container.querySelector('[role="status"]');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('aria-label', 'Loading tool selection');
    });
  });

  describe('CategoryChartSkeleton', () => {
    it('should render skeleton loader', () => {
      const { container } = render(<CategoryChartSkeleton />);
      const skeleton = container.querySelector('[role="status"]');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('aria-label', 'Loading category chart');
    });
  });

  describe('AttentionWeightsSkeleton', () => {
    it('should render skeleton loader', () => {
      const { container } = render(<AttentionWeightsSkeleton />);
      const skeleton = container.querySelector('[role="status"]');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('aria-label', 'Loading attention weights');
    });
  });

  describe('ThinkingStepsSkeleton', () => {
    it('should render skeleton loader', () => {
      const { container } = render(<ThinkingStepsSkeleton />);
      const skeleton = container.querySelector('[role="status"]');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('aria-label', 'Loading thinking steps');
    });
  });

  describe('ReasoningPanelSkeleton', () => {
    it('should render all skeleton components', () => {
      const { container } = render(<ReasoningPanelSkeleton />);
      const skeletons = container.querySelectorAll('[role="status"]');
      // Should have multiple skeleton loaders (tool, category, attention, thinking)
      expect(skeletons.length).toBeGreaterThan(1);
    });
  });

  describe('LoadingOverlay', () => {
    it('should render with default message', () => {
      render(<LoadingOverlay />);
      expect(screen.getByText('Yükleniyor...')).toBeInTheDocument();
    });

    it('should render with custom message', () => {
      render(<LoadingOverlay message="Öneriler hazırlanıyor..." />);
      expect(screen.getByText('Öneriler hazırlanıyor...')).toBeInTheDocument();
    });

    it('should have spinner', () => {
      const { container } = render(<LoadingOverlay />);
      const spinner = container.querySelector('[role="status"]');
      expect(spinner).toBeInTheDocument();
    });
  });
});

describe('ErrorStates', () => {
  describe('ErrorMessage', () => {
    it('should render error message', () => {
      render(<ErrorMessage error="Test error message" />);
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('should render with default title', () => {
      render(<ErrorMessage error="Test error" />);
      expect(screen.getByText('Bir Hata Oluştu')).toBeInTheDocument();
    });

    it('should render with custom title', () => {
      render(<ErrorMessage error="Test error" title="Custom Error" />);
      expect(screen.getByText('Custom Error')).toBeInTheDocument();
    });

    it('should render retry button when onRetry is provided', () => {
      const onRetry = vi.fn();
      render(<ErrorMessage error="Test error" onRetry={onRetry} />);
      const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
      expect(retryButton).toBeInTheDocument();
    });

    it('should not render retry button when onRetry is not provided', () => {
      render(<ErrorMessage error="Test error" />);
      const retryButton = screen.queryByRole('button', { name: /tekrar dene/i });
      expect(retryButton).not.toBeInTheDocument();
    });

    it('should call onRetry when retry button is clicked', async () => {
      const user = userEvent.setup();
      const onRetry = vi.fn();
      render(<ErrorMessage error="Test error" onRetry={onRetry} />);
      
      const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
      await user.click(retryButton);
      
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it('should have proper accessibility attributes', () => {
      const { container } = render(<ErrorMessage error="Test error" />);
      const alert = container.querySelector('[role="alert"]');
      expect(alert).toBeInTheDocument();
      expect(alert).toHaveAttribute('aria-live', 'assertive');
    });

    it('should render error icon', () => {
      const { container } = render(<ErrorMessage error="Test error" />);
      const icon = container.querySelector('svg[aria-hidden="true"]');
      expect(icon).toBeInTheDocument();
    });
  });

  describe('InlineErrorMessage', () => {
    it('should render inline error message', () => {
      render(<InlineErrorMessage message="Inline error" />);
      expect(screen.getByText('Inline error')).toBeInTheDocument();
    });

    it('should have role="alert"', () => {
      const { container } = render(<InlineErrorMessage message="Inline error" />);
      const alert = container.querySelector('[role="alert"]');
      expect(alert).toBeInTheDocument();
    });

    it('should have error styling', () => {
      const { container } = render(<InlineErrorMessage message="Inline error" />);
      const alert = container.querySelector('[role="alert"]');
      expect(alert?.className).toMatch(/text-red/);
    });
  });

  describe('ReasoningUnavailableError', () => {
    it('should render unavailable message', () => {
      render(<ReasoningUnavailableError />);
      expect(screen.getByText('Reasoning Bilgisi Mevcut Değil')).toBeInTheDocument();
    });

    it('should render close button when onClose is provided', () => {
      const onClose = vi.fn();
      render(<ReasoningUnavailableError onClose={onClose} />);
      const closeButton = screen.getByRole('button', { name: /kapat/i });
      expect(closeButton).toBeInTheDocument();
    });

    it('should call onClose when close button is clicked', async () => {
      const user = userEvent.setup();
      const onClose = vi.fn();
      render(<ReasoningUnavailableError onClose={onClose} />);
      
      const closeButton = screen.getByRole('button', { name: /kapat/i });
      await user.click(closeButton);
      
      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('NetworkError', () => {
    it('should render network error message', () => {
      render(<NetworkError />);
      expect(screen.getByText(/bağlantı hatası/i)).toBeInTheDocument();
      expect(screen.getByText((content, element) => {
        return element?.textContent === 'İnternet bağlantınızı kontrol edin ve tekrar deneyin.';
      })).toBeInTheDocument();
    });

    it('should render retry button when onRetry is provided', () => {
      const onRetry = vi.fn();
      render(<NetworkError onRetry={onRetry} />);
      const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
      expect(retryButton).toBeInTheDocument();
    });
  });

  describe('TimeoutError', () => {
    it('should render timeout error message', () => {
      render(<TimeoutError />);
      expect(screen.getAllByText(/zaman aşımı/i).length).toBeGreaterThan(0);
      expect(screen.getByText((content, element) => {
        return element?.textContent === 'İstek zaman aşımına uğradı. Lütfen tekrar deneyin.';
      })).toBeInTheDocument();
    });

    it('should render retry button when onRetry is provided', () => {
      const onRetry = vi.fn();
      render(<TimeoutError onRetry={onRetry} />);
      const retryButton = screen.getByRole('button', { name: /tekrar dene/i });
      expect(retryButton).toBeInTheDocument();
    });
  });

  describe('EmptyState', () => {
    it('should render with default title and message', () => {
      render(<EmptyState />);
      expect(screen.getByText('Sonuç Bulunamadı')).toBeInTheDocument();
      expect(screen.getByText(/aradığınız kriterlere uygun sonuç bulunamadı/i)).toBeInTheDocument();
    });

    it('should render with custom title and message', () => {
      render(<EmptyState title="Custom Title" message="Custom message" />);
      expect(screen.getByText('Custom Title')).toBeInTheDocument();
      expect(screen.getByText('Custom message')).toBeInTheDocument();
    });

    it('should render action button when provided', () => {
      const onAction = vi.fn();
      render(<EmptyState actionLabel="Try Again" onAction={onAction} />);
      const actionButton = screen.getByRole('button', { name: /try again/i });
      expect(actionButton).toBeInTheDocument();
    });

    it('should call onAction when action button is clicked', async () => {
      const user = userEvent.setup();
      const onAction = vi.fn();
      render(<EmptyState actionLabel="Try Again" onAction={onAction} />);
      
      const actionButton = screen.getByRole('button', { name: /try again/i });
      await user.click(actionButton);
      
      expect(onAction).toHaveBeenCalledTimes(1);
    });

    it('should not render action button when not provided', () => {
      render(<EmptyState />);
      const actionButton = screen.queryByRole('button');
      expect(actionButton).not.toBeInTheDocument();
    });
  });
});
