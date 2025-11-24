import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { ThinkingStep } from '@/types/reasoning';

export interface ThinkingStepsTimelineProps {
  steps: ThinkingStep[];
  onStepClick?: (step: ThinkingStep) => void;
  className?: string;
}

/**
 * Displays thinking steps in a vertical timeline with chronological ordering
 * 
 * @example
 * ```tsx
 * <ThinkingStepsTimeline
 *   steps={[
 *     { step: 1, action: 'Analyze user profile', result: 'Identified hobbies', insight: 'User prefers outdoor activities' },
 *     { step: 2, action: 'Filter categories', result: 'Selected 5 categories', insight: 'Sports and outdoor categories match' }
 *   ]}
 *   onStepClick={(step) => console.log('Clicked step:', step)}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable with Tab, Enter, and Space
 * - Chronological ordering for logical flow
 * - Scrollable container for long timelines
 */
export const ThinkingStepsTimeline: React.FC<ThinkingStepsTimelineProps> = ({
  steps,
  onStepClick,
  className,
}) => {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Sort steps chronologically
  const sortedSteps = [...steps].sort((a, b) => a.step - b.step);

  const handleStepClick = (step: ThinkingStep) => {
    setExpandedStep(expandedStep === step.step ? null : step.step);
    onStepClick?.(step);
  };

  const handleKeyDown = (e: React.KeyboardEvent, step: ThinkingStep) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleStepClick(step);
    }
  };

  // Auto-scroll to expanded step
  useEffect(() => {
    if (expandedStep !== null && timelineRef.current) {
      const expandedElement = timelineRef.current.querySelector(
        `[data-step="${expandedStep}"]`
      );
      if (expandedElement && typeof expandedElement.scrollIntoView === 'function') {
        expandedElement.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }
    }
  }, [expandedStep]);

  return (
    <div
      className={cn(
        'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
        className
      )}
      role="region"
      aria-label="Düşünme adımları zaman çizelgesi"
    >
      <h3 className="mb-4 text-lg font-semibold text-gray-900">
        Düşünme Adımları
      </h3>

      <div
        ref={timelineRef}
        className="max-h-[600px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100"
        role="list"
      >
        {sortedSteps.map((step, index) => {
          const isExpanded = expandedStep === step.step;
          const isLastStep = index === sortedSteps.length - 1;

          return (
            <div
              key={step.step}
              data-step={step.step}
              className={cn(
                'relative pb-8',
                isLastStep && 'pb-0'
              )}
              role="listitem"
            >
              {/* Timeline line */}
              {!isLastStep && (
                <div
                  className="absolute left-[15px] top-[30px] h-full w-0.5 bg-gray-200"
                  aria-hidden="true"
                />
              )}

              {/* Timeline item */}
              <div
                className={cn(
                  'relative flex gap-4 rounded-lg border p-4 transition-all cursor-pointer',
                  isExpanded
                    ? 'border-blue-300 bg-blue-50 shadow-md'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
                )}
                onClick={() => handleStepClick(step)}
                onKeyDown={(e) => handleKeyDown(e, step)}
                tabIndex={0}
                role="button"
                aria-expanded={isExpanded}
                aria-label={`Adım ${step.step}: ${step.action}`}
              >
                {/* Step marker with checkmark */}
                <div className="flex-shrink-0">
                  <div
                    className={cn(
                      'flex h-8 w-8 items-center justify-center rounded-full border-2 transition-colors',
                      'bg-green-500 border-green-600'
                    )}
                    aria-hidden="true"
                  >
                    {/* Green checkmark for completed steps */}
                    <svg
                      className="h-5 w-5 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2.5}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </div>
                </div>

                {/* Step content */}
                <div className="flex-1 min-w-0">
                  {/* Step number and action */}
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="inline-flex items-center justify-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-800">
                          Adım {step.step}
                        </span>
                        {isExpanded && (
                          <span className="text-xs text-gray-500">
                            (Detayları gizlemek için tıklayın)
                          </span>
                        )}
                      </div>
                      <h4 className="text-base font-semibold text-gray-900">
                        {step.action}
                      </h4>
                    </div>

                    {/* Expand/collapse indicator */}
                    <div
                      className={cn(
                        'flex-shrink-0 transition-transform',
                        isExpanded && 'rotate-180'
                      )}
                      aria-hidden="true"
                    >
                      <svg
                        className="h-5 w-5 text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 9l-7 7-7-7"
                        />
                      </svg>
                    </div>
                  </div>

                  {/* Result */}
                  <div className="mb-2">
                    <p className="text-sm text-gray-700">
                      <span className="font-medium text-gray-900">Sonuç:</span>{' '}
                      {step.result}
                    </p>
                  </div>

                  {/* Expanded details - Insight */}
                  {isExpanded && (
                    <div
                      className="mt-3 rounded-md bg-white border border-blue-200 p-3 animate-in fade-in slide-in-from-top-2 duration-200"
                      role="region"
                      aria-label="Adım detayları"
                    >
                      <div className="flex items-start gap-2">
                        <svg
                          className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                          />
                        </svg>
                        <div className="flex-1">
                          <p className="text-xs font-semibold text-blue-900 mb-1">
                            İçgörü:
                          </p>
                          <p className="text-sm text-gray-700">
                            {step.insight}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {sortedSteps.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          Düşünme adımı bilgisi mevcut değil
        </div>
      )}

      {/* Keyboard navigation hint */}
      {sortedSteps.length > 0 && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          Klavye ile gezinmek için Tab, Enter veya Space tuşlarını kullanın
        </div>
      )}
    </div>
  );
};
