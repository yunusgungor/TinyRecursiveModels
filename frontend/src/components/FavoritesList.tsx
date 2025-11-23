import { useEffect } from 'react';
import { useAppStore } from '@/store/useAppStore';
import type { GiftItem } from '@/lib/api/types';

interface FavoritesListProps {
  onGiftClick?: (gift: GiftItem) => void;
  onTrendyolClick?: (url: string) => void;
}

export function FavoritesList({ onGiftClick, onTrendyolClick }: FavoritesListProps) {
  const { favorites, loadFavorites, removeFavorite } = useAppStore();

  useEffect(() => {
    loadFavorites();
  }, [loadFavorites]);

  const handleRemove = (giftId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    removeFavorite(giftId);
  };

  const handleGiftClick = (gift: GiftItem) => {
    if (onGiftClick) {
      onGiftClick(gift);
    }
  };

  const handleTrendyolClick = (url: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (onTrendyolClick) {
      onTrendyolClick(url);
    } else {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Az önce';
    if (diffMins < 60) return `${diffMins} dakika önce`;
    if (diffHours < 24) return `${diffHours} saat önce`;
    if (diffDays < 7) return `${diffDays} gün önce`;
    
    return date.toLocaleDateString('tr-TR', {
      day: 'numeric',
      month: 'short',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
    });
  };

  if (favorites.length === 0) {
    return (
      <div className="text-center py-12">
        <svg
          className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600 mb-4"
          fill="none"
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
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Henüz Favori Eklemediniz
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Beğendiğiniz ürünleri favorilere ekleyerek daha sonra kolayca bulabilirsiniz
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Favorilerim ({favorites.length})
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {favorites.map((favorite) => (
          <div
            key={favorite.gift.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer overflow-hidden"
            onClick={() => handleGiftClick(favorite.gift)}
          >
            {/* Image */}
            <div className="relative h-48 bg-gray-200 dark:bg-gray-700">
              <img
                src={favorite.gift.imageUrl}
                alt={favorite.gift.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.src = 'https://via.placeholder.com/300x300?text=No+Image';
                }}
              />
              {!favorite.gift.inStock && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                  <span className="bg-red-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    Stokta Yok
                  </span>
                </div>
              )}
            </div>

            {/* Content */}
            <div className="p-4">
              <div className="flex items-start justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white line-clamp-2 flex-1">
                  {favorite.gift.name}
                </h3>
                <button
                  onClick={(e) => handleRemove(favorite.gift.id, e)}
                  className="ml-2 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 transition-colors flex-shrink-0"
                  aria-label="Favorilerden çıkar"
                >
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                  </svg>
                </button>
              </div>

              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {favorite.gift.category}
                </span>
                <span className="text-gray-400 dark:text-gray-600">•</span>
                <div className="flex items-center">
                  <svg className="w-4 h-4 text-yellow-400 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {favorite.gift.rating.toFixed(1)}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {favorite.gift.price.toFixed(2)} ₺
                </span>
                <button
                  onClick={(e) => handleTrendyolClick(favorite.gift.trendyolUrl, e)}
                  className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Trendyol'da Gör
                </button>
              </div>

              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Eklenme: {formatDate(favorite.addedAt)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
