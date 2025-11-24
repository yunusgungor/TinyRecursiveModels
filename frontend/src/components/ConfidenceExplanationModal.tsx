import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { cn } from '@/lib/utils/cn';
import { ConfidenceExplanation } from '@/types';

export interface ConfidenceExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: ConfidenceExplanation;
  className?: string;
}

/**
 * Displays a modal with detailed confidence explanation
 * Shows positive and negative factors that influenced the confidence score
 * 
 * @example
 * ```tsx
 * <ConfidenceExplanationModal
 *   isOpen={isModalOpen}
 *   onClose={() => setIsModalOpen(false)}
 *   explanation={{
 *     score: 0.85,
 *     level: 'high',
 *     factors: {
 *       positive: ['Perfect hobby match', 'Within budget'],
 *       negative: ['Limited availability']
 *     }
 *   }}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels and roles for screen readers
 * - Keyboard navigable with Tab, Enter, and Escape
 * - Focus trap within modal
 * - Escape key closes modal
 */
export const ConfidenceExplanationModal: React.FC<ConfidenceExplanationModalProps> = ({
  isOpen,
  onClose,
  explanation,
  className,
}) => {
  const getLevelLabel = (level: 'high' | 'medium' | 'low'): string => {
    switch (level) {
      case 'high':
        return 'Yüksek Güven';
      case 'medium':
        return 'Orta Güven';
      case 'low':
        return 'Düşük Güven';
    }
  };

  const getLevelColor = (level: 'high' | 'medium' | 'low'): string => {
    switch (level) {
      case 'high':
        return 'text-green-700';
      case 'medium':
        return 'text-yellow-700';
      case 'low':
        return 'text-red-700';
    }
  };

  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 z-50" />
        <Dialog.Content
          className={cn(
            'fixed left-[50%] top-[50%] z-50 w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 bg-white p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] rounded-lg',
            className
          )}
          aria-describedby="confidence-explanation-description"
        >
          <div className="flex flex-col space-y-4">
            {/* Header */}
            <Dialog.Title className="text-xl font-semibold text-gray-900">
              Güven Skoru Açıklaması
            </Dialog.Title>

            {/* Confidence Score Display */}
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex flex-col">
                <span className="text-sm text-gray-600">Güven Skoru</span>
                <span className="text-3xl font-bold text-gray-900">
                  {(explanation.score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-sm text-gray-600">Seviye</span>
                <span className={cn('text-lg font-semibold', getLevelColor(explanation.level))}>
                  {getLevelLabel(explanation.level)}
                </span>
              </div>
            </div>

            {/* Description */}
            <p id="confidence-explanation-description" className="text-sm text-gray-600">
              Bu güven skoru, hediye önerisinin ne kadar uygun olduğunu gösterir. 
              Aşağıda skoru etkileyen faktörleri görebilirsiniz.
            </p>

            {/* Positive Factors */}
            {explanation.factors.positive.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-green-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  Olumlu Faktörler
                </h3>
                <ul className="space-y-2" role="list" aria-label="Olumlu faktörler">
                  {explanation.factors.positive.map((factor, index) => (
                    <li
                      key={`positive-${index}`}
                      className="flex items-start gap-2 text-sm text-gray-700 bg-green-50 p-3 rounded-md border border-green-200"
                    >
                      <svg
                        className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Negative Factors */}
            {explanation.factors.negative.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                  <svg
                    className="w-5 h-5 text-red-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  Olumsuz Faktörler
                </h3>
                <ul className="space-y-2" role="list" aria-label="Olumsuz faktörler">
                  {explanation.factors.negative.map((factor, index) => (
                    <li
                      key={`negative-${index}`}
                      className="flex items-start gap-2 text-sm text-gray-700 bg-red-50 p-3 rounded-md border border-red-200"
                    >
                      <svg
                        className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Close Button */}
            <div className="flex justify-end pt-4 border-t border-gray-200">
              <Dialog.Close asChild>
                <button
                  type="button"
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                  onClick={onClose}
                >
                  Kapat
                </button>
              </Dialog.Close>
            </div>
          </div>

          {/* Close button (X) */}
          <Dialog.Close asChild>
            <button
              type="button"
              className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-white transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:pointer-events-none"
              aria-label="Kapat"
              onClick={onClose}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
