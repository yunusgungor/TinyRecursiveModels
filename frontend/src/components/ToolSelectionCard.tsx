import React from 'react';
import { cn } from '@/lib/utils/cn';
import { ToolSelectionReasoning } from '@/types/reasoning';
import * as Tooltip from '@radix-ui/react-tooltip';

export interface ToolSelectionCardProps {
  toolSelection: ToolSelectionReasoning[];
  className?: string;
}

/**
 * Displays tool selection reasoning with visual indicators
 * 
 * @example
 * ```tsx
 * <ToolSelectionCard
 *   toolSelection={[
 *     { name: 'review_analysis', selected: true, score: 0.85, reason: 'High rating match', confidence: 0.9, priority: 1 },
 *     { name: 'trend_analysis', selected: false, score: 0.45, reason: 'Low trend relevance', confidence: 0.4, priority: 2 }
 *   ]}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable tooltips
 * - Color-blind friendly with icons and text
 */
export const ToolSelectionCard: React.FC<ToolSelectionCardProps> = ({
  toolSelection,
  className,
}) => {
  // Sort tools by priority
  const sortedTools = [...toolSelection].sort((a, b) => a.priority - b.priority);

  const getToolDisplayName = (name: string): string => {
    const displayNames: Record<string, string> = {
      review_analysis: 'Yorum Analizi',
      trend_analysis: 'Trend Analizi',
      inventory_check: 'Stok Kontrolü',
      price_comparison: 'Fiyat Karşılaştırma',
      category_filter: 'Kategori Filtresi',
    };
    return displayNames[name] || name;
  };

  const formatFactors = (factors?: Record<string, number>): string => {
    if (!factors || Object.keys(factors).length === 0) {
      return 'Faktör bilgisi yok';
    }
    return Object.entries(factors)
      .map(([key, value]) => `${key}: ${(value * 100).toFixed(0)}%`)
      .join(', ');
  };

  return (
    <Tooltip.Provider>
      <div
        className={cn(
          'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
          className
        )}
        role="region"
        aria-label="Tool seçim bilgisi"
      >
        <h3 className="mb-4 text-lg font-semibold text-gray-900">
          Tool Seçimi
        </h3>

        <div className="space-y-3" role="list">
          {sortedTools.map((tool, index) => {
            const isLowConfidence = tool.confidence < 0.5;
            const displayName = getToolDisplayName(tool.name);

            return (
              <Tooltip.Root key={`${tool.name}-${index}`} delayDuration={200}>
                <Tooltip.Trigger asChild>
                  <div
                    className={cn(
                      'flex items-center justify-between rounded-md border p-3 transition-colors',
                      tool.selected
                        ? 'border-green-300 bg-green-50'
                        : 'border-gray-200 bg-gray-50',
                      'hover:shadow-md cursor-pointer'
                    )}
                    role="listitem"
                    aria-label={`${displayName}: ${tool.selected ? 'seçildi' : 'seçilmedi'}, güven skoru ${(tool.confidence * 100).toFixed(0)}%`}
                  >
                    <div className="flex items-center gap-3">
                      {/* Selection indicator */}
                      <div
                        className={cn(
                          'flex h-6 w-6 items-center justify-center rounded-full',
                          tool.selected
                            ? 'bg-green-500 text-white'
                            : 'bg-gray-300 text-gray-500'
                        )}
                        aria-hidden="true"
                      >
                        {tool.selected ? (
                          <svg
                            className="h-4 w-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        ) : (
                          <svg
                            className="h-4 w-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                        )}
                      </div>

                      {/* Tool name and priority */}
                      <div className="flex flex-col">
                        <span
                          className={cn(
                            'font-medium',
                            tool.selected ? 'text-green-900' : 'text-gray-700'
                          )}
                        >
                          {displayName}
                        </span>
                        <span className="text-xs text-gray-500">
                          Öncelik: {tool.priority}
                        </span>
                      </div>
                    </div>

                    {/* Confidence score */}
                    <div className="flex items-center gap-2">
                      <div className="text-right">
                        <div
                          className={cn(
                            'text-sm font-semibold',
                            tool.selected ? 'text-green-700' : 'text-gray-600'
                          )}
                        >
                          {(tool.confidence * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-gray-500">Güven</div>
                      </div>

                      {/* Low confidence warning */}
                      {isLowConfidence && (
                        <Tooltip.Root delayDuration={0}>
                          <Tooltip.Trigger asChild>
                            <div
                              className="flex h-5 w-5 items-center justify-center rounded-full bg-yellow-100 text-yellow-600"
                              aria-label="Düşük güven uyarısı"
                            >
                              <svg
                                className="h-3 w-3"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                              >
                                <path
                                  fillRule="evenodd"
                                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                                  clipRule="evenodd"
                                />
                              </svg>
                            </div>
                          </Tooltip.Trigger>
                          <Tooltip.Portal>
                            <Tooltip.Content
                              className="z-50 max-w-xs rounded-md bg-gray-900 px-3 py-2 text-sm text-white shadow-lg"
                              sideOffset={5}
                            >
                              <div className="font-semibold mb-1">Düşük Güven</div>
                              <div className="text-xs">
                                Bu tool düşük güven skoruna sahip. Sonuçlar güvenilir olmayabilir.
                              </div>
                              <Tooltip.Arrow className="fill-gray-900" />
                            </Tooltip.Content>
                          </Tooltip.Portal>
                        </Tooltip.Root>
                      )}
                    </div>
                  </div>
                </Tooltip.Trigger>

                {/* Main tooltip with reason and factors */}
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="z-50 max-w-sm rounded-md bg-gray-900 px-4 py-3 text-sm text-white shadow-lg"
                    sideOffset={5}
                  >
                    <div className="space-y-2">
                      <div>
                        <div className="font-semibold mb-1">Seçim Nedeni:</div>
                        <div className="text-xs">{tool.reason}</div>
                      </div>
                      
                      {tool.factors && Object.keys(tool.factors).length > 0 && (
                        <div>
                          <div className="font-semibold mb-1">Etkileyen Faktörler:</div>
                          <div className="text-xs">{formatFactors(tool.factors)}</div>
                        </div>
                      )}
                      
                      <div className="border-t border-gray-700 pt-2">
                        <div className="text-xs">
                          Skor: {(tool.score * 100).toFixed(0)}% | 
                          Güven: {(tool.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <Tooltip.Arrow className="fill-gray-900" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            );
          })}
        </div>

        {sortedTools.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            Tool seçim bilgisi mevcut değil
          </div>
        )}
      </div>
    </Tooltip.Provider>
  );
};
