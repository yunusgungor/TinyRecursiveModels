import React, { useRef, memo } from 'react';
import { cn } from '@/lib/utils/cn';
import { AttentionWeights, ChartType } from '@/types/reasoning';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { useIsMobile } from '@/hooks/useMediaQuery';
import { useOptimizedChart, useInViewport } from '@/hooks/useOptimizedChart';

export interface OptimizedAttentionWeightsChartProps {
  attentionWeights: AttentionWeights;
  chartType: ChartType;
  onChartTypeChange: (type: ChartType) => void;
  className?: string;
}

/**
 * Optimized version of AttentionWeightsChart with performance enhancements
 * - Debounced data updates
 * - Lazy rendering when in viewport
 * - Memoized chart data transformations
 * 
 * @example
 * ```tsx
 * <OptimizedAttentionWeightsChart
 *   attentionWeights={weights}
 *   chartType="bar"
 *   onChartTypeChange={setChartType}
 * />
 * ```
 */
const OptimizedAttentionWeightsChartComponent: React.FC<OptimizedAttentionWeightsChartProps> = ({
  attentionWeights,
  chartType,
  onChartTypeChange,
  className,
}) => {
  const isMobile = useIsMobile();
  const containerRef = useRef<HTMLDivElement>(null);
  const isInViewport = useInViewport(containerRef, { threshold: 0.1 });

  // Debounce data updates to prevent excessive re-renders
  const [optimizedWeights, isUpdating] = useOptimizedChart(attentionWeights, 300);

  // Memoize data transformations
  const userFeaturesData = React.useMemo(() => {
    return Object.entries(optimizedWeights.user_features).map(([name, value]) => ({
      name: formatFeatureName(name),
      value: value * 100,
      fullValue: value,
    }));
  }, [optimizedWeights.user_features]);

  const giftFeaturesData = React.useMemo(() => {
    return Object.entries(optimizedWeights.gift_features).map(([name, value]) => ({
      name: formatFeatureName(name),
      value: value * 100,
      fullValue: value,
    }));
  }, [optimizedWeights.gift_features]);

  function formatFeatureName(name: string): string {
    const nameMap: Record<string, string> = {
      hobbies: 'Hobiler',
      budget: 'Bütçe',
      age: 'Yaş',
      occasion: 'Özel Gün',
      category: 'Kategori',
      price: 'Fiyat',
      rating: 'Değerlendirme',
    };
    return nameMap[name] || name;
  }

  // Memoized custom tooltip
  const CustomTooltip = React.useCallback(({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="rounded-md bg-gray-900 px-3 py-2 text-sm text-white shadow-lg">
          <p className="font-semibold">{data.name}</p>
          <p>Ağırlık: {data.value.toFixed(1)}%</p>
          <p className="text-xs text-gray-300">({data.fullValue.toFixed(3)})</p>
        </div>
      );
    }
    return null;
  }, []);

  // Render placeholder when not in viewport
  if (!isInViewport) {
    return (
      <div
        ref={containerRef}
        className={cn(
          'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
          className
        )}
        style={{ minHeight: '600px' }}
      >
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-400">Grafik yükleniyor...</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'rounded-lg border border-gray-200 bg-white p-4 shadow-sm',
        isUpdating && 'opacity-75 transition-opacity',
        className
      )}
      role="region"
      aria-label="Attention weights bilgisi"
    >
      {/* Header with chart type toggle */}
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Attention Weights
        </h3>

        <div
          className="inline-flex rounded-md shadow-sm"
          role="group"
          aria-label="Grafik tipi seçimi"
        >
          <button
            type="button"
            onClick={() => onChartTypeChange('bar')}
            className={cn(
              'px-4 py-2 text-sm font-medium rounded-l-md border transition-colors',
              chartType === 'bar'
                ? 'bg-blue-600 text-white border-blue-600 z-10'
                : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
            )}
            aria-pressed={chartType === 'bar'}
            aria-label="Bar grafik"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => onChartTypeChange('radar')}
            className={cn(
              'px-4 py-2 text-sm font-medium rounded-r-md border border-l-0 transition-colors',
              chartType === 'radar'
                ? 'bg-blue-600 text-white border-blue-600 z-10'
                : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
            )}
            aria-pressed={chartType === 'radar'}
            aria-label="Radar grafik"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* User Features Chart */}
      <div className="mb-6">
        <h4 className="mb-3 text-sm font-semibold text-gray-700">
          Kullanıcı Özellikleri
        </h4>
        <div role="img" aria-label="Kullanıcı özellikleri attention weights grafiği">
          {chartType === 'bar' ? (
            <ResponsiveContainer width="100%" height={isMobile ? 220 : 250}>
              <BarChart
                data={userFeaturesData}
                margin={
                  isMobile
                    ? { top: 5, right: 10, left: 10, bottom: 40 }
                    : { top: 5, right: 30, left: 20, bottom: 5 }
                }
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="name"
                  angle={isMobile ? -45 : 0}
                  textAnchor={isMobile ? 'end' : 'middle'}
                  height={isMobile ? 60 : 30}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <YAxis
                  label={{
                    value: 'Ağırlık (%)',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                  domain={[0, 100]}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <ResponsiveContainer width="100%" height={isMobile ? 250 : 300}>
              <RadarChart data={userFeaturesData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" tick={{ fontSize: isMobile ? 10 : 12 }} />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <Radar
                  dataKey="value"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.6}
                />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Gift Features Chart */}
      <div>
        <h4 className="mb-3 text-sm font-semibold text-gray-700">
          Hediye Özellikleri
        </h4>
        <div role="img" aria-label="Hediye özellikleri attention weights grafiği">
          {chartType === 'bar' ? (
            <ResponsiveContainer width="100%" height={isMobile ? 220 : 250}>
              <BarChart
                data={giftFeaturesData}
                margin={
                  isMobile
                    ? { top: 5, right: 10, left: 10, bottom: 40 }
                    : { top: 5, right: 30, left: 20, bottom: 5 }
                }
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="name"
                  angle={isMobile ? -45 : 0}
                  textAnchor={isMobile ? 'end' : 'middle'}
                  height={isMobile ? 60 : 30}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <YAxis
                  label={{
                    value: 'Ağırlık (%)',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                  domain={[0, 100]}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <ResponsiveContainer width="100%" height={isMobile ? 250 : 300}>
              <RadarChart data={giftFeaturesData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" tick={{ fontSize: isMobile ? 10 : 12 }} />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: isMobile ? 10 : 12 }}
                />
                <Radar
                  dataKey="value"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.6}
                />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};

// Memoize the entire component
export const OptimizedAttentionWeightsChart = memo(
  OptimizedAttentionWeightsChartComponent,
  (prevProps, nextProps) => {
    return (
      JSON.stringify(prevProps.attentionWeights) === JSON.stringify(nextProps.attentionWeights) &&
      prevProps.chartType === nextProps.chartType
    );
  }
);

OptimizedAttentionWeightsChart.displayName = 'OptimizedAttentionWeightsChart';
