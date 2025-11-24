import { GiftRecommendation, ToolResults } from '@/lib/api/types';
import { useAppStore } from '@/store/useAppStore';
import { LazyImage } from './LazyImage';

interface RecommendationCardProps {
  recommendation: GiftRecommendation;
  toolResults?: ToolResults;
  onDetailsClick?: () => void;
  onTrendyolClick?: () => void;
}

export function RecommendationCard({
  recommendation,
  toolResults,
  onDetailsClick,
  onTrendyolClick,
}: RecommendationCardProps) {
  const { gift, confidenceScore } = recommendation;
  const isLowConfidence = confidenceScore < 0.5;
  const { isFavorite, addFavorite, removeFavorite } = useAppStore();
  const isFav = isFavorite(gift.id);

  const handleFavoriteToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isFav) {
      removeFavorite(gift.id);
    } else {
      addFavorite(gift);
    }
  };

  const handleTrendyolClick = () => {
    if (onTrendyolClick) {
      onTrendyolClick();
    } else {
      window.open(gift.trendyolUrl, '_blank', 'noopener,noreferrer');
    }
  };

  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('tr-TR', {
      style: 'currency',
      currency: 'TRY',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const renderStars = (rating: number) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    return (
      <div className="flex items-center gap-1">
        {[...Array(fullStars)].map((_, i) => (
          <svg
            key={`full-${i}`}
            className="w-4 h-4 text-yellow-400 fill-current"
            viewBox="0 0 20 20"
          >
            <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
          </svg>
        ))}
        {hasHalfStar && (
          <svg
            className="w-4 h-4 text-yellow-400"
            viewBox="0 0 20 20"
          >
            <defs>
              <linearGradient id="half">
                <stop offset="50%" stopColor="currentColor" />
                <stop offset="50%" stopColor="transparent" />
              </linearGradient>
            </defs>
            <path
              fill="url(#half)"
              d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z"
            />
          </svg>
        )}
        {[...Array(emptyStars)].map((_, i) => (
          <svg
            key={`empty-${i}`}
            className="w-4 h-4 text-gray-300 dark:text-gray-600"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
          </svg>
        ))}
        <span className="ml-1 text-sm text-gray-600 dark:text-gray-400">
          ({rating.toFixed(1)})
        </span>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col">
      {/* Low Confidence Warning */}
      {isLowConfidence && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 p-2 sm:p-3">
          <div className="flex items-start sm:items-center">
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400 mr-2 flex-shrink-0 mt-0.5 sm:mt-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-xs sm:text-sm text-yellow-800 dark:text-yellow-200">
              Bu öneri düşük güven skoruna sahip. Diğer seçenekleri de değerlendirmenizi öneririz.
            </p>
          </div>
        </div>
      )}

      {/* Product Image */}
      <div className="relative aspect-square w-full overflow-hidden bg-gray-100 dark:bg-gray-700">
        <LazyImage
          src={gift.imageUrl}
          alt={gift.name}
          className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
          placeholderClassName="aspect-square"
        />
        {!gift.inStock && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
            <span className="bg-red-600 text-white px-4 py-2 rounded-md font-semibold">
              Stokta Yok
            </span>
          </div>
        )}
        <div className="absolute top-2 right-2 flex gap-1.5 sm:gap-2">
          <button
            onClick={handleFavoriteToggle}
            className="bg-white dark:bg-gray-800 p-2 rounded-full shadow-md hover:scale-110 active:scale-95 transition-transform touch-target touch-manipulation"
            aria-label={isFav ? 'Favorilerden çıkar' : 'Favorilere ekle'}
          >
            <svg
              className={`w-4 h-4 sm:w-5 sm:h-5 ${isFav ? 'text-red-500 fill-current' : 'text-gray-400'}`}
              fill={isFav ? 'currentColor' : 'none'}
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
              />
            </svg>
          </button>
          <div className="bg-white dark:bg-gray-800 px-2 sm:px-3 py-1 rounded-full shadow-md">
            <span className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white">
              {Math.round(confidenceScore * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Product Info */}
      <div className="p-3 sm:p-4 flex flex-col flex-grow">
        {/* Category Badge */}
        <div className="mb-2">
          <span className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-medium px-2.5 py-0.5 rounded">
            {gift.category}
          </span>
        </div>

        {/* Product Name */}
        <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2 min-h-[2.5rem] sm:min-h-[3.5rem]">
          {gift.name}
        </h3>

        {/* Rating */}
        <div className="mb-3">
          {renderStars(gift.rating)}
        </div>

        {/* Price */}
        <div className="mb-4 flex-grow">
          <p className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
            {formatPrice(gift.price)}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col gap-2">
          {onDetailsClick && (
            <button
              onClick={onDetailsClick}
              className="w-full bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white py-2.5 sm:py-2 px-4 rounded-md font-medium hover:bg-gray-200 active:bg-gray-300 dark:hover:bg-gray-600 dark:active:bg-gray-500 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 touch-target touch-manipulation"
            >
              Detaylar
            </button>
          )}
          <button
            onClick={handleTrendyolClick}
            disabled={!gift.inStock}
            className="w-full bg-orange-500 text-white py-2.5 sm:py-2 px-4 rounded-md font-medium hover:bg-orange-600 active:bg-orange-700 transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed touch-target touch-manipulation"
          >
            Trendyol'da Gör
          </button>
        </div>
      </div>
    </div>
  );
}
