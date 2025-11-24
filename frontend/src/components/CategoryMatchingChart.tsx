import React, { useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { CategoryMatchingReasoning } from '@/types/reasoning';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { useIsMobile } from '@/hooks/useMediaQuery';

export interface CategoryMatchingChartProps {
  categories: CategoryMatchingReasoning[];
  onCategoryClick?: (category: CategoryMatchingReasoning) => void;
  className?: string;
}

/**
 * Displays category matching scores with visual indicators
 * 
 * @example
 * ```tsx
 * <CategoryMatchingChart
 *   categories={[
 *     { category_name: 'Elektronik', score: 0.85, reasons: ['Hobi eşleşmesi'], feature_contributions: { hobby: 0.9 } },
 *     { category_name: 'Kitap', score: 0.65, reasons: ['Yaş uygunluğu'], feature_contributions: { age: 0.7 } }
 *   ]}
 *   onCategoryClick={(cat) => console.log(cat)}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable
 * - Color-blind friendly with patterns and text
 */
export const CategoryMatchingChart: React.FC<CategoryMatchingChartProps> = ({
  categories,
  onCategoryClick,
  className,
}) => {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const isMobile = useIsMobile();

  // Sort categories by score (descending) and take at least top 3
  const sortedCategories = [...categories]
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(3, categories.length));

  // Get color based on score
  const getScoreColor = (score: number): string => {
    if (score > 0.7) return '#10b981'; // green-500
    if (score < 0.3) return '#ef4444'; // red-500
    return '#eab308'; // yellow-500
  };

  // Get color class for text based on score
  const getScoreTextColor = (score: number): string => {
    if (score > 0.7) return 'text-green-700';
    if (score < 0.3) return 'text-red-700';
    return 'text-yellow-700';
  };

  // Format score as percentage
  const formatScorePercentage = (score: number): string => {
    return `${(score * 100).toFixed(0)}%`;
  };

  // Prepare data for Recharts
  const chartData = sortedCategories.map((cat) => ({
    name: cat.category_name,
    score: cat.score * 100, // Convert to percentage for display
    rawScore: cat.score,
    color: getScoreColor(cat.score),
  }));

  const handleCategoryClick = (category: CategoryMatchingReasoning) => {
    setExpandedCategory(
      expandedCategory === category.category_name ? null : category.category_name
    );
    onCategoryClick?.(category);
  };

  const formatFeatureContributions = (contributions: Record<string, number>): string => {
    if (!contributions || Object.keys(contributions).length === 0) {
      return 'Katkı bilgisi yok';
    }
    return Object.entries(contributions)
      .map(([key, value]) => `${key}: ${(value * 100).toFixed(0)}%`)
      .join(', ');
  };

  return (
    <TooltipPrimitive.Provider>
      <div
        className={cn(
          'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
          className
        )}
        role="region"
        aria-label="Kategori eşleştirme bilgisi"
      >
        <h3 className="mb-4 text-lg font-semibold text-gray-900">
          Kategori Eşleştirme
        </h3>

        {sortedCategories.length > 0 ? (
          <>
            {/* Bar Chart */}
            <div className="mb-4" role="img" aria-label="Kategori skorları bar grafiği">
              <ResponsiveContainer width="100%" height={isMobile ? 250 : 300}>
                <BarChart
                  data={chartData}
                  layout={isMobile ? 'horizontal' : 'vertical'}
                  margin={isMobile 
                    ? { top: 5, right: 30, left: 20, bottom: 60 }
                    : { top: 5, right: 30, left: 100, bottom: 5 }
                  }
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  {isMobile ? (
                    <>
                      <XAxis 
                        type="category" 
                        dataKey="name" 
                        angle={-45}
                        textAnchor="end"
                        height={80}
                      />
                      <YAxis 
                        type="number"
                        domain={[0, 100]}
                        label={{ value: 'Skor (%)', angle: -90, position: 'insideLeft' }}
                      />
                    </>
                  ) : (
                    <>
                      <XAxis
                        type="number"
                        domain={[0, 100]}
                        label={{ value: 'Skor (%)', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis type="category" dataKey="name" />
                    </>
                  )}
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="rounded-md bg-gray-900 px-3 py-2 text-sm text-white shadow-lg">
                            <p className="font-semibold">{data.name}</p>
                            <p>Skor: {data.score.toFixed(1)}%</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="score" radius={isMobile ? [4, 4, 0, 0] : [0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Category Details List */}
            <div className="space-y-3" role="list">
              {sortedCategories.map((category, index) => {
                const isExpanded = expandedCategory === category.category_name;

                return (
                  <div
                    key={`${category.category_name}-${index}`}
                    className={cn(
                      'rounded-md border p-3 transition-all cursor-pointer hover:shadow-md',
                      category.score > 0.7
                        ? 'border-green-300 bg-green-50'
                        : category.score < 0.3
                        ? 'border-red-300 bg-red-50'
                        : 'border-yellow-300 bg-yellow-50'
                    )}
                    role="listitem"
                    onClick={() => handleCategoryClick(category)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleCategoryClick(category);
                      }
                    }}
                    tabIndex={0}
                    aria-expanded={isExpanded}
                    aria-label={`${category.category_name}: ${formatScorePercentage(category.score)} skor`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {/* Score indicator */}
                        <div
                          className={cn(
                            'flex h-10 w-10 items-center justify-center rounded-full font-bold text-white',
                            category.score > 0.7
                              ? 'bg-green-500'
                              : category.score < 0.3
                              ? 'bg-red-500'
                              : 'bg-yellow-500'
                          )}
                          aria-hidden="true"
                        >
                          {formatScorePercentage(category.score)}
                        </div>

                        {/* Category name */}
                        <div className="flex flex-col">
                          <span
                            className={cn(
                              'font-medium',
                              getScoreTextColor(category.score)
                            )}
                          >
                            {category.category_name}
                          </span>
                          <span className="text-xs text-gray-500">
                            {category.reasons.length} neden
                          </span>
                        </div>
                      </div>

                      {/* Expand indicator */}
                      <div
                        className={cn(
                          'transition-transform',
                          isExpanded && 'rotate-180'
                        )}
                        aria-hidden="true"
                      >
                        <svg
                          className="h-5 w-5 text-gray-500"
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

                    {/* Expanded details */}
                    {isExpanded && (
                      <div className="mt-3 space-y-2 border-t pt-3">
                        <div>
                          <div className="text-sm font-semibold text-gray-700 mb-1">
                            Eşleştirme Nedenleri:
                          </div>
                          <ul className="list-disc list-inside space-y-1">
                            {category.reasons.map((reason, idx) => (
                              <li key={idx} className="text-sm text-gray-600">
                                {reason}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {category.feature_contributions &&
                          Object.keys(category.feature_contributions).length > 0 && (
                            <div>
                              <div className="text-sm font-semibold text-gray-700 mb-1">
                                Özellik Katkıları:
                              </div>
                              <div className="text-sm text-gray-600">
                                {formatFeatureContributions(category.feature_contributions)}
                              </div>
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            Kategori eşleştirme bilgisi mevcut değil
          </div>
        )}
      </div>
    </TooltipPrimitive.Provider>
  );
};
