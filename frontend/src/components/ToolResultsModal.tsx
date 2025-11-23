import { GiftRecommendation, ToolResults } from '@/lib/api/types';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ToolResultsModalProps {
  gift: GiftRecommendation;
  toolResults?: ToolResults;
  isOpen: boolean;
  onClose: () => void;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export function ToolResultsModal({
  gift,
  toolResults,
  isOpen,
  onClose,
}: ToolResultsModalProps) {
  if (!isOpen) return null;

  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('tr-TR', {
      style: 'currency',
      currency: 'TRY',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const renderPriceComparison = () => {
    if (!toolResults?.priceComparison) return null;

    const { bestPrice, averagePrice, priceRange, savingsPercentage, checkedPlatforms } =
      toolResults.priceComparison;

    const priceData = [
      { name: 'En İyi Fiyat', value: bestPrice },
      { name: 'Ortalama Fiyat', value: averagePrice },
      { name: 'Mevcut Fiyat', value: gift.gift.price },
    ];

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Fiyat Karşılaştırması
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">En İyi Fiyat</p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {formatPrice(bestPrice)}
            </p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Tasarruf</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">
              %{savingsPercentage.toFixed(1)}
            </p>
          </div>
        </div>
        <div className="mb-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Fiyat Aralığı: <span className="font-semibold">{priceRange}</span>
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Kontrol Edilen Platformlar: {checkedPlatforms.join(', ')}
          </p>
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value) => formatPrice(value as number)} />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderReviewAnalysis = () => {
    if (!toolResults?.reviewAnalysis) return null;

    const {
      averageRating,
      totalReviews,
      sentimentScore,
      keyPositives,
      keyNegatives,
      recommendationConfidence,
    } = toolResults.reviewAnalysis;

    const sentimentData = [
      { name: 'Pozitif', value: sentimentScore * 100 },
      { name: 'Negatif', value: (1 - sentimentScore) * 100 },
    ];

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Yorum Analizi
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Ortalama Puan</p>
            <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
              {averageRating.toFixed(1)} / 5.0
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Toplam Yorum</p>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {totalReviews.toLocaleString('tr-TR')}
            </p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Güven Skoru</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">
              %{(recommendationConfidence * 100).toFixed(0)}
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              Olumlu Yönler
            </h4>
            <ul className="list-disc list-inside space-y-1">
              {keyPositives.map((positive, index) => (
                <li key={index} className="text-sm text-gray-700 dark:text-gray-300">
                  {positive}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              Olumsuz Yönler
            </h4>
            <ul className="list-disc list-inside space-y-1">
              {keyNegatives.map((negative, index) => (
                <li key={index} className="text-sm text-gray-700 dark:text-gray-300">
                  {negative}
                </li>
              ))}
            </ul>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={sentimentData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: %${value.toFixed(1)}`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {sentimentData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `%${(value as number).toFixed(1)}`} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderTrendAnalysis = () => {
    if (!toolResults?.trendAnalysis) return null;

    const { trendDirection, popularityScore, growthRate, trendingItems } =
      toolResults.trendAnalysis;

    const trendData = [
      { name: 'Popülerlik', value: popularityScore * 100 },
      { name: 'Büyüme Oranı', value: growthRate * 100 },
    ];

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Trend Analizi
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Trend Yönü</p>
            <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
              {trendDirection === 'up' ? '↑ Yükseliyor' : trendDirection === 'down' ? '↓ Düşüyor' : '→ Stabil'}
            </p>
          </div>
          <div className="bg-pink-50 dark:bg-pink-900/20 p-4 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Popülerlik Skoru</p>
            <p className="text-2xl font-bold text-pink-600 dark:text-pink-400">
              {(popularityScore * 100).toFixed(0)}/100
            </p>
          </div>
        </div>
        {trendingItems.length > 0 && (
          <div className="mb-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              Trend Ürünler
            </h4>
            <div className="flex flex-wrap gap-2">
              {trendingItems.map((item, index) => (
                <span
                  key={index}
                  className="bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-3 py-1 rounded-full text-sm"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value) => `%${(value as number).toFixed(1)}`} />
            <Bar dataKey="value" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderBudgetOptimizer = () => {
    if (!toolResults?.budgetOptimizer) return null;

    const { recommendedAllocation, valueScore, savingsOpportunities } =
      toolResults.budgetOptimizer;

    const allocationData = Object.entries(recommendedAllocation).map(([name, value]) => ({
      name,
      value,
    }));

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Bütçe Optimizasyonu
        </h3>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg mb-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">Değer Skoru</p>
          <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
            {(valueScore * 100).toFixed(0)}/100
          </p>
        </div>
        {savingsOpportunities.length > 0 && (
          <div className="mb-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
              Tasarruf Fırsatları
            </h4>
            <ul className="list-disc list-inside space-y-1">
              {savingsOpportunities.map((opportunity, index) => (
                <li key={index} className="text-sm text-gray-700 dark:text-gray-300">
                  {opportunity}
                </li>
              ))}
            </ul>
          </div>
        )}
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={allocationData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${formatPrice(value)}`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {allocationData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => formatPrice(value as number)} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderInventoryCheck = () => {
    if (!toolResults?.inventoryCheck) return null;

    const { inStock, stockLevel, estimatedRestockDate } = toolResults.inventoryCheck;

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Stok Durumu
        </h3>
        <div className={`p-4 rounded-lg ${inStock ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}>
          <div className="flex items-center mb-2">
            <div className={`w-3 h-3 rounded-full mr-2 ${inStock ? 'bg-green-500' : 'bg-red-500'}`} />
            <p className={`text-lg font-semibold ${inStock ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
              {inStock ? 'Stokta Var' : 'Stokta Yok'}
            </p>
          </div>
          {stockLevel && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Stok Seviyesi: <span className="font-semibold">{stockLevel}</span>
            </p>
          )}
          {estimatedRestockDate && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Tahmini Yeniden Stok Tarihi: <span className="font-semibold">{estimatedRestockDate}</span>
            </p>
          )}
        </div>
      </div>
    );
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black bg-opacity-50 p-0 sm:p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white dark:bg-gray-800 rounded-t-2xl sm:rounded-lg shadow-xl max-w-4xl w-full max-h-[95vh] sm:max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 sm:p-6 flex justify-between items-start z-10">
          <div className="flex-1 pr-2">
            <h2 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-white mb-2 line-clamp-2">
              {gift.gift.name}
            </h2>
            <div className="flex flex-wrap items-center gap-2 sm:gap-4">
              <span className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                {formatPrice(gift.gift.price)}
              </span>
              <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs sm:text-sm font-medium px-2.5 py-0.5 rounded">
                {gift.gift.category}
              </span>
              <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                Güven: %{(gift.confidenceScore * 100).toFixed(0)}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="ml-2 sm:ml-4 text-gray-400 hover:text-gray-600 active:text-gray-800 dark:hover:text-gray-200 dark:active:text-white transition-colors touch-target touch-manipulation flex-shrink-0"
            aria-label="Kapat"
          >
            <svg
              className="w-6 h-6"
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
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-6">
          {/* Reasoning */}
          {gift.reasoning && gift.reasoning.length > 0 && (
            <div className="mb-4 sm:mb-6">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-3">
                Öneri Gerekçesi
              </h3>
              <ul className="space-y-2">
                {gift.reasoning.map((reason, index) => (
                  <li key={index} className="flex items-start">
                    <svg
                      className="w-4 h-4 sm:w-5 sm:h-5 text-green-500 mr-2 mt-0.5 flex-shrink-0"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span className="text-sm sm:text-base text-gray-700 dark:text-gray-300">{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Tool Results */}
          {toolResults && (
            <>
              {renderInventoryCheck()}
              {renderPriceComparison()}
              {renderReviewAnalysis()}
              {renderTrendAnalysis()}
              {renderBudgetOptimizer()}
            </>
          )}

          {!toolResults && (
            <div className="text-center py-8 text-sm sm:text-base text-gray-500 dark:text-gray-400">
              Detaylı analiz sonuçları mevcut değil.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 p-4 sm:p-6">
          <div className="flex flex-col gap-2 sm:gap-3">
            <button
              onClick={onClose}
              className="w-full bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white py-3 px-4 rounded-md font-medium hover:bg-gray-300 active:bg-gray-400 dark:hover:bg-gray-600 dark:active:bg-gray-500 transition-colors touch-target touch-manipulation"
            >
              Kapat
            </button>
            <button
              onClick={() => window.open(gift.gift.trendyolUrl, '_blank', 'noopener,noreferrer')}
              disabled={!gift.gift.inStock}
              className="w-full bg-orange-500 text-white py-3 px-4 rounded-md font-medium hover:bg-orange-600 active:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed touch-target touch-manipulation"
            >
              Trendyol'da Gör
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
