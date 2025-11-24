/**
 * Error state components for reasoning visualization
 * Provides user-friendly error messages with retry functionality
 */

import React from 'react';

export interface ErrorMessageProps {
  error: Error | string;
  onRetry?: () => void;
  title?: string;
  className?: string;
}

/**
 * Generic error message component with retry button
 * 
 * @example
 * ```tsx
 * <ErrorMessage
 *   error={error}
 *   onRetry={refetch}
 *   title="Öneriler Yüklenemedi"
 * />
 * ```
 */
export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  error,
  onRetry,
  title = 'Bir Hata Oluştu',
  className = '',
}) => {
  const errorMessage = typeof error === 'string' ? error : error.message;

  return (
    <div
      className={`bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 ${className}`}
      role="alert"
      aria-live="assertive"
    >
      <div className="flex items-start space-x-3">
        {/* Error icon */}
        <svg
          className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>

        <div className="flex-1">
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
            {title}
          </h3>
          <p className="text-red-700 dark:text-red-300 mb-4">{errorMessage}</p>

          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              aria-label="Tekrar dene"
            >
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Tekrar Dene
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Compact error message for inline display
 */
export const InlineErrorMessage: React.FC<{ message: string; className?: string }> = ({
  message,
  className = '',
}) => {
  return (
    <div
      className={`flex items-center space-x-2 text-red-600 dark:text-red-400 text-sm ${className}`}
      role="alert"
    >
      <svg
        className="w-4 h-4 flex-shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      <span>{message}</span>
    </div>
  );
};

/**
 * Error state for reasoning panel when data is unavailable
 */
export const ReasoningUnavailableError: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
  return (
    <div
      className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6"
      role="alert"
    >
      <div className="flex items-start space-x-3">
        <svg
          className="w-6 h-6 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>

        <div className="flex-1">
          <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
            Reasoning Bilgisi Mevcut Değil
          </h3>
          <p className="text-yellow-700 dark:text-yellow-300 mb-4">
            Bu öneri için detaylı reasoning bilgisi bulunmamaktadır. Lütfen daha sonra tekrar
            deneyin veya başka bir öneriyi inceleyin.
          </p>

          {onClose && (
            <button
              onClick={onClose}
              className="inline-flex items-center px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2"
              aria-label="Kapat"
            >
              Kapat
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Error boundary fallback component
 */
export const ErrorBoundaryFallback: React.FC<{
  error: Error;
  resetErrorBoundary: () => void;
}> = ({ error, resetErrorBoundary }) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
      <div className="max-w-md w-full">
        <ErrorMessage
          error={error}
          onRetry={resetErrorBoundary}
          title="Beklenmeyen Bir Hata Oluştu"
        />
      </div>
    </div>
  );
};

/**
 * Network error specific component
 */
export const NetworkError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => {
  return (
    <ErrorMessage
      error="İnternet bağlantınızı kontrol edin ve tekrar deneyin."
      onRetry={onRetry}
      title="Bağlantı Hatası"
    />
  );
};

/**
 * API timeout error component
 */
export const TimeoutError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => {
  return (
    <ErrorMessage
      error="İstek zaman aşımına uğradı. Lütfen tekrar deneyin."
      onRetry={onRetry}
      title="Zaman Aşımı"
    />
  );
};

/**
 * Empty state component (not an error, but related)
 */
export const EmptyState: React.FC<{
  title?: string;
  message?: string;
  actionLabel?: string;
  onAction?: () => void;
}> = ({
  title = 'Sonuç Bulunamadı',
  message = 'Aradığınız kriterlere uygun sonuç bulunamadı.',
  actionLabel,
  onAction,
}) => {
  return (
    <div className="text-center py-12">
      <svg
        className="mx-auto h-12 w-12 text-gray-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
        />
      </svg>
      <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-gray-100">{title}</h3>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{message}</p>
      {actionLabel && onAction && (
        <div className="mt-6">
          <button
            onClick={onAction}
            className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            {actionLabel}
          </button>
        </div>
      )}
    </div>
  );
};
