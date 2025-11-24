/**
 * RecommendationsPage - Main page for displaying gift recommendations with reasoning
 * 
 * This page integrates:
 * - useRecommendations hook for fetching recommendations
 * - GiftRecommendationCard for displaying individual recommendations
 * - ReasoningPanel for detailed reasoning analysis
 * - ComparisonView for side-by-side comparison
 * - Loading and error states
 * 
 * Requirements: 1.1, 7.1, 7.2
 */

import React, { useState, useEffect } from 'react';
import { useRecommendations } from '@/hooks/useRecommendations';
import { useReasoningContext } from '@/contexts/ReasoningContext';
import { GiftRecommendationCard } from '@/components/GiftRecommendationCard';
import { ReasoningPanel } from '@/components/ReasoningPanel';
import { ComparisonView } from '@/components/ComparisonView';
import { GiftCardSkeleton } from '@/components/LoadingStates';
import { ErrorMessage } from '@/components/ErrorStates';
import type { UserProfile, EnhancedGiftRecommendation } from '@/types/reasoning';

export interface RecommendationsPageProps {
  userProfile: UserProfile;
  onBackToSearch?: () => void;
}

/**
 * RecommendationsPage component
 * 
 * Displays gift recommendations with reasoning information
 * Supports detailed reasoning panel and comparison mode
 * 
 * @example
 * ```tsx
 * <RecommendationsPage
 *   userProfile={profile}
 *   onBackToSearch={() => navigate('/search')}
 * />
 * ```
 */
export const RecommendationsPage: React.FC<RecommendationsPageProps> = ({
  userProfile,
  onBackToSearch,
}) => {
  // Reasoning context for state management
  const {
    reasoningLevel,
    selectedGifts,
    toggleGiftSelection,
    clearSelection,
    isComparisonMode,
    setComparisonMode,
    isPanelOpen,
    openPanel,
    closePanel,
    activeFilters,
    setFilters,
    chartType,
    setChartType,
  } = useReasoningContext();

  // Fetch recommendations
  const {
    recommendations,
    toolResults,
    isLoading,
    error,
    refetch,
    cancel,
  } = useRecommendations({
    userProfile,
    includeReasoning: true,
    reasoningLevel,
    maxRecommendations: 5,
  });

  // Selected recommendation for detailed panel
  const [selectedRecommendation, setSelectedRecommendation] = useState<EnhancedGiftRecommendation | null>(null);

  // Fetch recommendations on mount
  useEffect(() => {
    refetch();
    
    // Cleanup: cancel request on unmount
    return () => {
      cancel();
    };
  }, [userProfile, reasoningLevel]);

  // Handle show details button click
  const handleShowDetails = (recommendation: EnhancedGiftRecommendation) => {
    setSelectedRecommendation(recommendation);
    openPanel();
  };

  // Handle panel close
  const handleClosePanel = () => {
    closePanel();
    setSelectedRecommendation(null);
  };

  // Handle gift selection for comparison
  const handleGiftSelect = (giftId: string) => {
    toggleGiftSelection(giftId);
  };

  // Handle enter comparison mode
  const handleEnterComparisonMode = () => {
    if (selectedGifts.length >= 2) {
      setComparisonMode(true);
    }
  };

  // Handle exit comparison mode
  const handleExitComparisonMode = () => {
    setComparisonMode(false);
    clearSelection();
  };

  // Get selected recommendations for comparison
  const selectedRecommendations = recommendations.filter((rec) =>
    selectedGifts.includes(rec.gift.id)
  );

  // Render loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
        <div className="container mx-auto px-4">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Hediye Önerileri
            </h1>
            <p className="text-gray-600 dark:text-gray-400">Öneriler yükleniyor...</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <GiftCardSkeleton />
            <GiftCardSkeleton />
            <GiftCardSkeleton />
          </div>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
        <div className="container mx-auto px-4">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Hediye Önerileri
            </h1>
          </div>
          <ErrorMessage
            error={error}
            onRetry={refetch}
            title="Öneriler Yüklenemedi"
          />
          {onBackToSearch && (
            <div className="mt-4 text-center">
              <button
                onClick={onBackToSearch}
                className="px-6 py-3 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                Yeni Arama Yap
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Render comparison mode
  if (isComparisonMode && selectedRecommendations.length >= 2) {
    return (
      <ComparisonView
        recommendations={selectedRecommendations}
        onExit={handleExitComparisonMode}
      />
    );
  }

  // Render recommendations
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Hediye Önerileri
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                {recommendations.length} öneri bulundu
              </p>
            </div>

            {onBackToSearch && (
              <button
                onClick={onBackToSearch}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Aramaya geri dön"
              >
                Yeni Arama
              </button>
            )}
          </div>

          {/* Comparison Mode Button */}
          {selectedGifts.length >= 2 && (
            <div className="flex items-center justify-center">
              <button
                onClick={handleEnterComparisonMode}
                className="px-6 py-3 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 flex items-center gap-2"
                aria-label={`${selectedGifts.length} hediyeyi karşılaştır`}
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
                {selectedGifts.length} Hediyeyi Karşılaştır
              </button>
            </div>
          )}
        </div>

        {/* Empty State */}
        {recommendations.length === 0 && (
          <div className="max-w-2xl mx-auto">
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
                  {onBackToSearch && (
                    <button
                      onClick={onBackToSearch}
                      className="mt-4 px-4 py-2 bg-yellow-600 text-white rounded-md font-medium hover:bg-yellow-700 transition-colors"
                    >
                      Yeni Arama Yap
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations Grid */}
        {recommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map((recommendation) => (
              <GiftRecommendationCard
                key={recommendation.gift.id}
                recommendation={recommendation}
                toolResults={toolResults}
                onShowDetails={() => handleShowDetails(recommendation)}
                isSelected={selectedGifts.includes(recommendation.gift.id)}
                onSelect={() => handleGiftSelect(recommendation.gift.id)}
              />
            ))}
          </div>
        )}

        {/* Reasoning Panel */}
        {selectedRecommendation && (
          <ReasoningPanel
            isOpen={isPanelOpen}
            onClose={handleClosePanel}
            reasoningTrace={selectedRecommendation.reasoning_trace!}
            gift={selectedRecommendation.gift}
            userProfile={userProfile}
            activeFilters={activeFilters}
            onFilterChange={setFilters}
            chartType={chartType}
            onChartTypeChange={setChartType}
          />
        )}
      </div>
    </div>
  );
};
