import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface ConfidenceIndicatorProps {
  confidence: number;
  onClick?: () => void;
  className?: string;
}

/**
 * Displays a confidence indicator with color coding and label
 * 
 * @example
 * ```tsx
 * <ConfidenceIndicator
 *   confidence={0.85}
 *   onClick={() => openExplanationModal()}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable with Tab and Enter
 * - Color-blind friendly with text labels
 */
export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  onClick,
  className,
}) => {
  // Determine confidence level and styling
  const getConfidenceLevel = (): 'high' | 'medium' | 'low' => {
    if (confidence > 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
  };

  const level = getConfidenceLevel();

  const getConfidenceLabel = (): string => {
    switch (level) {
      case 'high':
        return 'Yüksek Güven';
      case 'medium':
        return 'Orta Güven';
      case 'low':
        return 'Düşük Güven';
    }
  };

  const getConfidenceColor = (): string => {
    switch (level) {
      case 'high':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'low':
        return 'bg-red-100 text-red-800 border-red-300';
    }
  };

  const label = getConfidenceLabel();
  const colorClasses = getConfidenceColor();

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onClick && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-medium transition-colors',
        colorClasses,
        onClick && 'cursor-pointer hover:opacity-80',
        className
      )}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      tabIndex={onClick ? 0 : undefined}
      role={onClick ? 'button' : 'status'}
      aria-label={`Güven skoru: ${(confidence * 100).toFixed(0)}%, ${label}`}
      aria-live="polite"
    >
      <span className="font-semibold">{(confidence * 100).toFixed(0)}%</span>
      <span>{label}</span>
    </div>
  );
};
