import React, { useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { EnhancedGiftRecommendation } from '@/types/reasoning';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { LazyImage } from './LazyImage';

export interface GiftRecommendationCardProps {
  recommendation: EnhancedGiftRecommendation;
  toolResults?: Record<string, any>;
  onShowDetails?: () => void;
  isSelected?: boolean;
  onSelect?: () => void;
  className?: string;
}

/**
 * Displays a gift recommendation card with reasoning information
 * 
 * @example
 * ```tsx
 * <GiftRecommendationCard
 *   recommendation={recommendation}
 *   toolResults={toolResults}
 *   onShowDetails={() => openPanel()}
 *   isSelected={false}
 *   onSelect={() => toggleSelection()}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable with Tab and Enter
 * - Color-blind friendly with icons
 */
export const GiftRecommendationCard: React.FC<GiftRecommendationCardProps> = ({
  recommendation,
  toolResults,
  onShowDetails,
  isSelected,
  onSelect,
  className,
}) => {
  const { gift, reasoning, confidence } = recommendation;
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Determine if reasoning text should be expandable
  const maxReasoningLength = 200;
  const reasoningText = reasoning.join(' ');
  const shouldShowExpandButton = reasoningText.length > maxReasoningLength;
  
  // Format price in Turkish Lira
  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('tr-TR', {
      style: 'currency',
      currency: 'TRY',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  // Highlight reasoning factors (hobi, bütçe, yaş)
  const highlightReasoningFactors = (text: string): React.ReactNode => {
    const keywords = [
      { pattern: /hobi|ilgi alanı|ilgi/gi, className: 'text-purple-600 dark:text-purple-400 font-medium' },
      { pattern: /bütçe|fiyat|uygun/gi, className: 'text-green-600 dark:text-green-400 font-medium' },
      { pattern: /yaş|yaşa uygun/gi, className: 'text-blue-600 dark:text-blue-400 font-medium' },
    ];

    let parts: React.ReactNode[] = [text];
    
    keywords.forEach(({ pattern, className }) => {
      parts = parts.flatMap((part) => {
        if (typeof part !== 'string') return part;
        
        const matches = [...part.matchAll(pattern)];
        if (matches.length === 0) return part;
        
        const result: React.ReactNode[] = [];
        let lastIndex = 0;
        
        matches.forEach((match, idx) => {
          const matchIndex = match.index!;
          if (matchIndex > lastIndex) {
            result.push(part.substring(lastIndex, matchIndex));
          }
          result.push(
            <span key={`${matchIndex}-${idx}`} className={className}>
              {match[0]}
            </span>
          );
          lastIndex = matchIndex + match[0].length;
        });
        
        if (lastIndex < part.length) {
          result.push(part.substring(lastIndex));
        }
        
        return result;
      });
    });
    
    return <>{parts}</>;
  };

  // Render tool insights as icons
  const renderToolInsights = () => {
    if (!toolResults) return null;

    return (
      <div className="flex items-center gap-2 mt-3" role="list" aria-label="Tool insights">
        {toolResults.review_analysis && (
          <div
            className="flex items-center gap-1 text-yellow-500"
            role="listitem"
            aria-label={`Rating: ${toolResults.review_analysis.average_rating}/5.0`}
            title={`Rating: ${toolResults.review_analysis.average_rating}/5.0`}
          >
            <svg
              className="w-5 h-5"
              fill="currentColor"
              viewBox="0 0 20 20"
              aria-hidden="true"
            >
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            <span className="text-sm font-medium">
              {toolResults.review_analysis.average_rating.toFixed(1)}
            </span>
          </div>
        )}
        
        {toolResults.trend_analysis?.trending && (
          <div
            className="flex items-center gap-1 text-green-500"
            role="listitem"
            aria-label="Trending"
            title="Trending"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
              />
            </svg>
            <span className="text-sm font-medium">Trend</span>
          </div>
        )}
        
        {toolResults.inventory_check?.available && (
          <div
            className="flex items-center gap-1 text-blue-500"
            role="listitem"
            aria-label="In Stock"
            title="In Stock"
          >
            <svg
              className="w-5 h-5"
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
            <span className="text-sm font-medium">Stokta</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={cn(
        'bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col',
        isSelected && 'ring-2 ring-blue-500',
        className
      )}
      role="article"
      aria-label={`Gift recommendation: ${gift.name}`}
    >
      {/* Product Image */}
      <div className="relative aspect-square w-full overflow-hidden bg-gray-100 dark:bg-gray-700">
        <LazyImage
          src={gift.image_url || ''}
          alt={gift.name}
          className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
          placeholderClassName="aspect-square"
        />
        
        {/* Confidence Indicator Overlay */}
        <div className="absolute top-2 right-2">
          <ConfidenceIndicator confidence={confidence} />
        </div>
      </div>

      {/* Product Info */}
      <div className="p-4 flex flex-col flex-grow">
        {/* Category Badge */}
        <div className="mb-2">
          <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium px-2.5 py-0.5 rounded">
            {gift.category}
          </span>
        </div>

        {/* Product Name */}
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2 min-h-[3.5rem]">
          {gift.name}
        </h3>

        {/* Price */}
        <div className="mb-3">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatPrice(gift.price)}
          </p>
        </div>

        {/* Reasoning Section */}
        {reasoning && reasoning.length > 0 && (
          <div
            className="mb-3 flex-grow"
            role="region"
            aria-label="Reasoning information"
          >
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              {reasoning.slice(0, isExpanded ? undefined : 2).map((text, idx) => (
                <p key={idx} className="leading-relaxed">
                  {highlightReasoningFactors(text)}
                </p>
              ))}
            </div>
            
            {shouldShowExpandButton && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="mt-2 text-sm text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
                aria-expanded={isExpanded}
                aria-label={isExpanded ? 'Daha az göster' : 'Daha fazla göster'}
              >
                {isExpanded ? 'Daha az göster' : 'Daha fazla göster'}
              </button>
            )}
          </div>
        )}

        {/* Tool Insights */}
        {renderToolInsights()}

        {/* Action Buttons */}
        <div className="flex flex-col gap-2 mt-4">
          {onShowDetails && (
            <button
              onClick={onShowDetails}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md font-medium hover:bg-blue-700 active:bg-blue-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              aria-label="Detaylı analiz göster"
            >
              Detaylı Analiz Göster
            </button>
          )}
          
          {onSelect && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={onSelect}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                aria-label="Karşılaştırma için seç"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Karşılaştırma için seç
              </span>
            </label>
          )}
        </div>
      </div>
    </div>
  );
};
