import React, { useRef, useState } from 'react';
import { cn } from '@/lib/utils/cn';
import type { ThinkingStep } from '@/types/reasoning';

export interface VirtualThinkingStepsTimelineProps {
  steps: ThinkingStep[];
  onStepClick?: (step: ThinkingStep) => void;
  className?: string;
  itemHeight?: number;
  containerHeight?: number;
}

/**
 * Virtual scrolling implementation for ThinkingStepsTimeline
 * Renders only visible items for better performance with long lists
 * 
 * @example
 * ```tsx
 * <VirtualThinkingStepsTimeline
 *   steps={thinkingSteps}
 *   onStepClick={handleStepClick}
 *   itemHeight={120}
 *   containerHeight={400}
 * />
 * ```
 */
export const VirtualThinkingStepsTimeline: React.FC<VirtualThinkingStepsTimelineProps> = ({
  steps,
  onStepClick,
  className,
  itemHeight = 120,
  containerHeight = 400,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  // Calculate visible range
  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(
    steps.length - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight)
  );

  // Get visible items with buffer
  const buffer = 2;
  const visibleStartIndex = Math.max(0, startIndex - buffer);
  const visibleEndIndex = Math.min(steps.length - 1, endIndex + buffer);
  const visibleSteps = steps.slice(visibleStartIndex, visibleEndIndex + 1);

  // Handle scroll
  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  // Handle step click
  const handleStepClick = (step: ThinkingStep) => {
    setExpandedStep(expandedStep === step.step ? null : step.step);
    onStepClick?.(step);
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent, step: ThinkingStep) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleStepClick(step);
    }
  };

  return (
    <div
      className={cn(
        'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
        className
      )}
      role="region"
      aria-label="Düşünme adımları timeline"
    >
      <h3 className="mb-4 text-lg font-semibold text-gray-900">
        Düşünme Adımları
      </h3>

      <div
        ref={containerRef}
        className="relative overflow-y-auto"
        style={{ height: `${containerHeight}px` }}
        onScroll={handleScroll}
        role="list"
      >
        {/* Spacer for total height */}
        <div style={{ height: `${steps.length * itemHeight}px`, position: 'relative' }}>
          {/* Render visible items */}
          {visibleSteps.map((step, index) => {
            const actualIndex = visibleStartIndex + index;
            const isExpanded = expandedStep === step.step;
            const actualHeight = isExpanded ? itemHeight * 1.5 : itemHeight;

            return (
              <div
                key={step.step}
                className="absolute left-0 right-0 transition-all duration-200"
                style={{
                  top: `${actualIndex * itemHeight}px`,
                  height: `${actualHeight}px`,
                }}
                role="listitem"
              >
                <div
                  className={cn(
                    'flex gap-4 p-4 cursor-pointer hover:bg-gray-50 rounded-lg transition-colors',
                    isExpanded && 'bg-blue-50'
                  )}
                  onClick={() => handleStepClick(step)}
                  onKeyDown={(e) => handleKeyDown(e, step)}
                  tabIndex={0}
                  aria-expanded={isExpanded}
                >
                  {/* Timeline marker */}
                  <div className="flex flex-col items-center">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100">
                      <svg
                        className="h-5 w-5 text-green-600"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="mt-1 text-xs font-medium text-gray-500">
                      #{step.step}
                    </div>
                    {actualIndex < steps.length - 1 && (
                      <div className="mt-1 h-full w-0.5 bg-gray-200" />
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-semibold text-gray-900 mb-1">
                      {step.action}
                    </h4>
                    <p className="text-sm text-gray-600 mb-1">
                      {step.result}
                    </p>

                    {isExpanded && (
                      <div className="mt-2 rounded-md bg-blue-50 p-3 animate-fadeIn">
                        <p className="text-sm text-gray-700">
                          <span className="font-semibold">Insight:</span> {step.insight}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Expand indicator */}
                  <div className="flex items-center">
                    <svg
                      className={cn(
                        'h-5 w-5 text-gray-400 transition-transform',
                        isExpanded && 'rotate-180'
                      )}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
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
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer info */}
      <div className="mt-4 text-sm text-gray-500">
        Toplam {steps.length} adım
      </div>
    </div>
  );
};
