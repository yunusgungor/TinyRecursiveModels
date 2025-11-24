import React from 'react';
import { cn } from '@/lib/utils/cn';
import { EnhancedGiftRecommendation } from '@/types/reasoning';
import { CategoryMatchingChart } from './CategoryMatchingChart';
import { AttentionWeightsChart } from './AttentionWeightsChart';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { LazyImage } from './LazyImage';

export interface ComparisonViewProps {
  recommendations: EnhancedGiftRecommendation[];
  onExit: () => void;
  className?: string;
}

/**
 * Displays side-by-side comparison of selected gift recommendations
 * 
 * @example
 * ```tsx
 * <ComparisonView
 *   recommendations={selectedRecommendations}
 *   onExit={() => exitComparisonMode()}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable
 * - Comparison charts are accessible
 */
export const ComparisonView: React.FC<ComparisonViewProps> = ({
  recommendations,
  onExit,
  className,
}) => {
  const [chartType, setChartType] = React.useState<'bar' | 'radar'>('bar');

  // Format price in Turkish Lira
  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('tr-TR', {
      style: 'currency',
      currency: 'TRY',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  // Prepare category comparison data
  const prepareCategoryComparisonData = () => {
    if (recommendations.length === 0) return [];

    // Get all unique categories from all recommendations
    const allCategories = new Set<string>();
    recommendations.forEach((rec) => {
      rec.reasoning_trace?.category_matching.forEach((cat) => {
        allCategories.add(cat.category_name);
      });
    });

    // Create comparison data for each category
    return Array.from(allCategories).map((categoryName) => {
      const dataPoint: any = { category: categoryName };
      
      recommendations.forEach((rec, idx) => {
        const categoryData = rec.reasoning_trace?.category_matching.find(
          (cat) => cat.category_name === categoryName
        );
        dataPoint[`gift${idx + 1}`] = categoryData ? categoryData.score * 100 : 0;
      });
      
      return dataPoint;
    });
  };

  const categoryComparisonData = prepareCategoryComparisonData();

  return (
    <div
      className={cn('comparison-view bg-white dark:bg-gray-900 min-h-screen p-6', className)}
      role="region"
      aria-label="Hediye karşılaştırma görünümü"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Hediye Karşılaştırma
        </h2>
        <button
          onClick={onExit}
          className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-label="Karşılaştırma modundan çık"
        >
          Karşılaştırmayı Kapat
        </button>
      </div>

      {/* Side-by-side gift cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {recommendations.map((recommendation, idx) => (
          <div
            key={recommendation.gift.id}
            className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-4"
            role="article"
            aria-label={`Hediye ${idx + 1}: ${recommendation.gift.name}`}
          >
            {/* Gift Image */}
            <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-700 mb-4">
              <LazyImage
                src={recommendation.gift.image_url || ''}
                alt={recommendation.gift.name}
                className="w-full h-full object-cover"
                placeholderClassName="aspect-square"
              />
              <div className="absolute top-2 right-2">
                <ConfidenceIndicator confidence={recommendation.confidence} />
              </div>
            </div>

            {/* Gift Info */}
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white line-clamp-2">
                {recommendation.gift.name}
              </h3>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {formatPrice(recommendation.gift.price)}
              </p>
              <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium px-2.5 py-0.5 rounded">
                {recommendation.gift.category}
              </span>
            </div>

            {/* Basic Reasoning */}
            <div className="mt-4 space-y-1">
              {recommendation.reasoning.slice(0, 2).map((text, textIdx) => (
                <p key={textIdx} className="text-sm text-gray-700 dark:text-gray-300">
                  • {text}
                </p>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Charts */}
      <div className="space-y-8">
        {/* Category Scores Comparison */}
        {categoryComparisonData.length > 0 && (
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Kategori Skorları Karşılaştırması
            </h3>
            <div className="overflow-x-auto">
              <CategoryComparisonChart
                data={categoryComparisonData}
                giftNames={recommendations.map((rec) => rec.gift.name)}
              />
            </div>
          </div>
        )}

        {/* Attention Weights Comparison */}
        {recommendations.every((rec) => rec.reasoning_trace?.attention_weights) && (
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Attention Weights Karşılaştırması
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {recommendations.map((recommendation, idx) => (
                <div key={recommendation.gift.id}>
                  <h4 className="text-md font-medium text-gray-800 dark:text-gray-200 mb-2">
                    {recommendation.gift.name}
                  </h4>
                  {recommendation.reasoning_trace?.attention_weights && (
                    <AttentionWeightsChart
                      attentionWeights={recommendation.reasoning_trace.attention_weights}
                      chartType={chartType}
                      onChartTypeChange={(type) => setChartType(type)}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Confidence Comparison Table */}
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Güven Skoru Karşılaştırması
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Hediye
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Güven Skoru
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Fiyat
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {recommendations.map((recommendation) => (
                  <tr key={recommendation.gift.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                      {recommendation.gift.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                      <div className="flex items-center gap-2">
                        <ConfidenceIndicator confidence={recommendation.confidence} />
                        <span>{(recommendation.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                      {formatPrice(recommendation.gift.price)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Category comparison chart component for side-by-side comparison
 */
interface CategoryComparisonChartProps {
  data: any[];
  giftNames: string[];
}

const CategoryComparisonChart: React.FC<CategoryComparisonChartProps> = ({ data, giftNames }) => {
  // Colors for different gifts
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="w-full" style={{ minHeight: '400px' }}>
      {data.map((categoryData, idx) => (
        <div key={idx} className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {categoryData.category}
            </span>
          </div>
          <div className="space-y-2">
            {giftNames.map((name, giftIdx) => {
              const score = categoryData[`gift${giftIdx + 1}`] || 0;
              const color = colors[giftIdx % colors.length];
              
              return (
                <div key={giftIdx} className="flex items-center gap-2">
                  <div className="w-32 text-xs text-gray-600 dark:text-gray-400 truncate">
                    {name}
                  </div>
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative">
                    <div
                      className="h-6 rounded-full flex items-center justify-end pr-2 text-xs font-medium text-white transition-all duration-300"
                      style={{
                        width: `${score}%`,
                        backgroundColor: color,
                      }}
                    >
                      {score > 10 && `${score.toFixed(0)}%`}
                    </div>
                  </div>
                  {score <= 10 && (
                    <span className="text-xs text-gray-600 dark:text-gray-400 w-12">
                      {score.toFixed(0)}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};
