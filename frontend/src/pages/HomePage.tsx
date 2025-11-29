import { useState, useEffect } from 'react';
import { UserProfileForm } from '@/components/UserProfileForm';
import { RecommendationCard } from '@/components/RecommendationCard';
import { ToolResultsModal } from '@/components/ToolResultsModal';
import { FavoritesList } from '@/components/FavoritesList';
import { SearchHistoryList } from '@/components/SearchHistoryList';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useRecommendations } from '@/hooks/useRecommendations';
import { useAppStore } from '@/store/useAppStore';
import type { UserProfile, GiftRecommendation, ToolResults } from '@/lib/api/types';

type ViewMode = 'search' | 'favorites' | 'history';

export function HomePage() {
  const [selectedRecommendation, setSelectedRecommendation] = useState<GiftRecommendation | null>(null);
  const [toolResults, setToolResults] = useState<ToolResults | undefined>(undefined);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('search');
  const [data, setData] = useState<{ recommendations: GiftRecommendation[]; inferenceTime: number; cacheHit: boolean } | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [isError, setIsError] = useState(false);
  const [error, setError] = useState<any>(null);

  const { addSearchHistory } = useAppStore();

  const handleSubmit = async (profile: UserProfile) => {
    // Add to search history
    addSearchHistory(profile);

    setIsPending(true);
    setIsError(false);
    setError(null);

    try {
      // Import the API client
      const { recommendationsApi } = await import('@/lib/api/recommendations');
      const response = await recommendationsApi.getRecommendations({
        userProfile: profile,
        maxRecommendations: 10,
        useCache: true,
      });

      setData(response);
      setToolResults(response.toolResults as ToolResults);
      setIsPending(false);
    } catch (err) {
      setError(err);
      setIsError(true);
      setIsPending(false);
    }
  };

  const handleHistoryProfileSelect = (profile: UserProfile) => {
    setViewMode('search');
    handleSubmit(profile);
  };

  const handleDetailsClick = (recommendation: GiftRecommendation) => {
    setSelectedRecommendation(recommendation);
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedRecommendation(null);
  };

  const handleTrendyolClick = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const renderLoadingState = () => (
    <div className="flex flex-col items-center justify-center py-12 sm:py-16 px-4">
      <div className="relative w-20 h-20 sm:w-24 sm:h-24 mb-4 sm:mb-6">
        {/* Spinning loader */}
        <div className="absolute inset-0 border-4 border-blue-200 dark:border-blue-900 rounded-full"></div>
        <div className="absolute inset-0 border-4 border-blue-600 dark:border-blue-400 rounded-full border-t-transparent animate-spin"></div>
      </div>
      <h3 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-white mb-2 text-center">
        Öneriler Hazırlanıyor...
      </h3>
      <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400 text-center max-w-md px-4">
        Yapay zeka modelimiz sizin için en uygun hediye önerilerini araştırıyor. Bu işlem birkaç saniye sürebilir.
      </p>
      {/* Progress dots animation */}
      <div className="flex gap-2 mt-4 sm:mt-6">
        <div className="w-3 h-3 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-3 h-3 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-3 h-3 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
    </div>
  );

  const renderErrorState = () => {
    let errorMessage = 'Bir hata oluştu. Lütfen tekrar deneyin.';
    let errorDetails = '';

    if (error) {
      // Handle different error types
      if ('response' in error && error.response) {
        const responseError = error.response as { data?: { message?: string; errorCode?: string } };
        errorMessage = responseError.data?.message || errorMessage;
        errorDetails = responseError.data?.errorCode || '';
      } else if ('message' in error) {
        errorMessage = (error as { message: string }).message;
      }
    }

    return (
      <div className="max-w-2xl mx-auto mt-8">
        <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-6 rounded-lg">
          <div className="flex items-start">
            <svg
              className="w-6 h-6 text-red-500 mr-3 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
                Öneri Alınamadı
              </h3>
              <p className="text-red-700 dark:text-red-300 mb-2">{errorMessage}</p>
              {errorDetails && (
                <p className="text-sm text-red-600 dark:text-red-400">
                  Hata Kodu: {errorDetails}
                </p>
              )}
              <button
                onClick={() => window.location.reload()}
                className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md font-medium hover:bg-red-700 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              >
                Sayfayı Yenile
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderRecommendations = () => {
    if (!data || !data.recommendations || data.recommendations.length === 0) {
      return (
        <div className="max-w-2xl mx-auto mt-8">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-6 rounded-lg">
            <div className="flex items-start">
              <svg
                className="w-6 h-6 text-yellow-500 mr-3 flex-shrink-0 mt-0.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
                  Öneri Bulunamadı
                </h3>
                <p className="text-yellow-700 dark:text-yellow-300">
                  Belirttiğiniz kriterlere uygun hediye bulunamadı. Lütfen farklı kriterler ile tekrar deneyin.
                </p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-8 sm:mt-12">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 gap-4">
            <div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Sizin İçin Seçtiklerimiz
              </h2>
              <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
                {data.recommendations.length} öneri bulundu
                {data.cacheHit && ' (Önbellekten)'}
                {' • '}
                İşlem süresi: {data.inferenceTime?.toFixed(2) || data.inference_time?.toFixed(2) || '0.00'}s
              </p>
            </div>
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 active:text-blue-800 dark:active:text-blue-200 font-medium flex items-center justify-center sm:justify-start gap-2 touch-target"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Yeni Arama
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            {data.recommendations.map((recommendation) => (
              <RecommendationCard
                key={recommendation.gift.id}
                recommendation={recommendation}
                toolResults={toolResults}
                onDetailsClick={() => handleDetailsClick(recommendation)}
                onTrendyolClick={() => handleTrendyolClick(recommendation.gift.trendyolUrl)}
              />
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-4 sm:py-6 md:py-8">
      {/* Theme Toggle Button */}
      <ThemeToggle />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-6 sm:mb-8">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-3">
            Trendyol Hediye Önerisi
          </h1>
          <p className="text-base sm:text-lg text-gray-600 dark:text-gray-400 px-4">
            Sevdikleriniz için mükemmel hediyeyi bulun
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center mb-6 sm:mb-8 overflow-x-auto">
          <div className="inline-flex rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-1 touch-manipulation">
            <button
              onClick={() => setViewMode('search')}
              className={`px-4 sm:px-6 py-2 rounded-md font-medium transition-colors touch-target whitespace-nowrap ${viewMode === 'search'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 active:bg-gray-200 dark:active:bg-gray-600'
                }`}
            >
              Arama
            </button>
            <button
              onClick={() => setViewMode('favorites')}
              className={`px-4 sm:px-6 py-2 rounded-md font-medium transition-colors touch-target whitespace-nowrap ${viewMode === 'favorites'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 active:bg-gray-200 dark:active:bg-gray-600'
                }`}
            >
              Favoriler
            </button>
            <button
              onClick={() => setViewMode('history')}
              className={`px-4 sm:px-6 py-2 rounded-md font-medium transition-colors touch-target whitespace-nowrap ${viewMode === 'history'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 active:bg-gray-200 dark:active:bg-gray-600'
                }`}
            >
              Geçmiş
            </button>
          </div>
        </div>

        {/* Search View */}
        {viewMode === 'search' && (
          <>
            {/* Form - Only show if no results yet */}
            {!data && !isPending && !isError && (
              <UserProfileForm onSubmit={handleSubmit} isLoading={isPending} />
            )}

            {/* Loading State */}
            {isPending && renderLoadingState()}

            {/* Error State */}
            {isError && renderErrorState()}

            {/* Recommendations */}
            {data && !isPending && !isError && renderRecommendations()}

            {/* Tool Results Modal */}
            {selectedRecommendation && (
              <ToolResultsModal
                gift={selectedRecommendation}
                toolResults={toolResults}
                isOpen={isModalOpen}
                onClose={handleModalClose}
              />
            )}
          </>
        )}

        {/* Favorites View */}
        {viewMode === 'favorites' && (
          <div className="max-w-7xl mx-auto">
            <FavoritesList
              onGiftClick={(gift) => {
                setSelectedRecommendation({
                  gift,
                  confidenceScore: 1,
                  reasoning: [],
                  toolInsights: {},
                  rank: 1,
                });
                setIsModalOpen(true);
              }}
              onTrendyolClick={(url) => window.open(url, '_blank', 'noopener,noreferrer')}
            />
          </div>
        )}

        {/* History View */}
        {viewMode === 'history' && (
          <div className="max-w-4xl mx-auto">
            <SearchHistoryList onProfileSelect={handleHistoryProfileSelect} />
          </div>
        )}
      </div>
    </div>
  );
}
