import { useEffect } from 'react';
import { useAppStore } from '@/store/useAppStore';
import type { UserProfile } from '@/lib/api/types';

interface SearchHistoryListProps {
  onProfileSelect?: (profile: UserProfile) => void;
}

export function SearchHistoryList({ onProfileSelect }: SearchHistoryListProps) {
  const { searchHistory, loadSearchHistory, removeSearchHistory, clearSearchHistory } = useAppStore();

  useEffect(() => {
    loadSearchHistory();
  }, [loadSearchHistory]);

  const handleRemove = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    removeSearchHistory(id);
  };

  const handleClearAll = () => {
    if (window.confirm('Tüm arama geçmişini silmek istediğinizden emin misiniz?')) {
      clearSearchHistory();
    }
  };

  const handleProfileClick = (profile: UserProfile) => {
    if (onProfileSelect) {
      onProfileSelect(profile);
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

  if (searchHistory.length === 0) {
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
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Henüz Arama Yapmadınız
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Yaptığınız aramalar burada görünecek
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Arama Geçmişi ({searchHistory.length})
        </h2>
        {searchHistory.length > 0 && (
          <button
            onClick={handleClearAll}
            className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 text-sm font-medium transition-colors"
          >
            Tümünü Temizle
          </button>
        )}
      </div>

      <div className="space-y-3">
        {searchHistory.map((item) => (
          <div
            key={item.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer p-4"
            onClick={() => handleProfileClick(item.profile)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <svg
                    className="w-5 h-5 text-gray-400 dark:text-gray-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {formatDate(item.timestamp)}
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
                      Yaş
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {item.profile.age}
                    </span>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
                      İlişki
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {item.profile.relationship}
                    </span>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
                      Bütçe
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {item.profile.budget.toFixed(2)} ₺
                    </span>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
                      Özel Gün
                    </span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {item.profile.occasion}
                    </span>
                  </div>

                  <div className="col-span-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">
                      Hobiler
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {item.profile.hobbies.slice(0, 3).map((hobby, idx) => (
                        <span
                          key={idx}
                          className="inline-block bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs px-2 py-1 rounded"
                        >
                          {hobby}
                        </span>
                      ))}
                      {item.profile.hobbies.length > 3 && (
                        <span className="inline-block bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs px-2 py-1 rounded">
                          +{item.profile.hobbies.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={(e) => handleRemove(item.id, e)}
                className="ml-4 text-gray-400 hover:text-red-600 dark:text-gray-600 dark:hover:text-red-400 transition-colors flex-shrink-0"
                aria-label="Geçmişten sil"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
