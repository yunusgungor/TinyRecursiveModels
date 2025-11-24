import React, { useState, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils/cn';
import type {
  ReasoningTrace,
  GiftItem,
  UserProfile,
  ReasoningFilter,
  ChartType,
  ExportFormat,
} from '@/types/reasoning';
import { ToolSelectionCard } from './ToolSelectionCard';
import { CategoryMatchingChart } from './CategoryMatchingChart';
import { AttentionWeightsChart } from './AttentionWeightsChart';
import { ThinkingStepsTimeline } from './ThinkingStepsTimeline';
import * as Dialog from '@radix-ui/react-dialog';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { useMediaQuery } from '@/hooks/useMediaQuery';

export interface ReasoningPanelProps {
  isOpen: boolean;
  onClose: () => void;
  reasoningTrace: ReasoningTrace;
  gift: GiftItem;
  userProfile: UserProfile;
  activeFilters: ReasoningFilter[];
  onFilterChange: (filters: ReasoningFilter[]) => void;
  chartType?: ChartType;
  onChartTypeChange?: (type: ChartType) => void;
  className?: string;
}

/**
 * Detailed reasoning panel component
 * Displays comprehensive reasoning information with filtering and export capabilities
 * 
 * @example
 * ```tsx
 * <ReasoningPanel
 *   isOpen={isOpen}
 *   onClose={handleClose}
 *   reasoningTrace={trace}
 *   gift={giftData}
 *   userProfile={profile}
 *   activeFilters={filters}
 *   onFilterChange={setFilters}
 * />
 * ```
 * 
 * @accessibility
 * - Full keyboard navigation support
 * - ARIA labels and roles
 * - Screen reader compatible
 * - Focus management
 */
export const ReasoningPanel: React.FC<ReasoningPanelProps> = ({
  isOpen,
  onClose,
  reasoningTrace,
  gift,
  userProfile,
  activeFilters,
  onFilterChange,
  chartType = 'bar',
  onChartTypeChange,
  className,
}) => {
  const isMobile = useMediaQuery('(max-width: 767px)');
  const panelRef = useRef<HTMLDivElement>(null);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  // Minimum swipe distance (in px)
  const minSwipeDistance = 50;

  // Handle swipe gestures for mobile
  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientY);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientY);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isDownSwipe = distance < -minSwipeDistance;
    
    if (isDownSwipe) {
      onClose();
    }
    
    setTouchStart(null);
    setTouchEnd(null);
  };

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Filter toggle handlers
  const toggleFilter = (filter: ReasoningFilter) => {
    if (activeFilters.includes(filter)) {
      onFilterChange(activeFilters.filter(f => f !== filter));
    } else {
      onFilterChange([...activeFilters, filter]);
    }
  };

  const showAll = () => {
    onFilterChange(['tool_selection', 'category_matching', 'attention_weights', 'thinking_steps']);
  };

  const clearAll = () => {
    onFilterChange([]);
  };

  // Export handlers
  const handleExport = (format: ExportFormat) => {
    switch (format) {
      case 'json':
        exportAsJSON();
        break;
      case 'pdf':
        exportAsPDF();
        break;
      case 'share':
        copyShareLink();
        break;
    }
  };

  const exportAsJSON = () => {
    const data = {
      gift,
      reasoning_trace: reasoningTrace,
      user_profile: userProfile,
      exported_at: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reasoning-${gift.id}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Show success message (you can replace with toast notification)
    console.log('JSON exported successfully');
  };

  const exportAsPDF = async () => {
    // TODO: Implement PDF export using jsPDF
    console.log('PDF export not yet implemented');
  };

  const copyShareLink = async () => {
    const link = `${window.location.origin}/recommendations/${gift.id}?reasoning=true`;
    
    try {
      await navigator.clipboard.writeText(link);
      console.log('Link copied to clipboard');
    } catch (error) {
      console.error('Failed to copy link:', error);
    }
  };

  // Check if a section should be displayed
  const shouldShowSection = (filter: ReasoningFilter): boolean => {
    return activeFilters.includes(filter);
  };

  const content = (
    <div
      ref={panelRef}
      className={cn(
        'reasoning-panel flex flex-col',
        isMobile ? 'h-full' : 'max-h-[90vh]',
        className
      )}
      onTouchStart={isMobile ? onTouchStart : undefined}
      onTouchMove={isMobile ? onTouchMove : undefined}
      onTouchEnd={isMobile ? onTouchEnd : undefined}
      role="region"
      aria-label="Detaylı reasoning analizi"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 p-4 md:p-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900 md:text-2xl">
            Detaylı Analiz
          </h2>
          <p className="mt-1 text-sm text-gray-600">
            {gift.name} için reasoning bilgileri
          </p>
        </div>

        <button
          onClick={onClose}
          className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-label="Paneli kapat"
        >
          <svg
            className="h-6 w-6"
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

      {/* Filter Bar */}
      <div className="border-b border-gray-200 bg-gray-50 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Filtreler:</span>
          
          <button
            onClick={() => toggleFilter('tool_selection')}
            className={cn(
              'rounded-full px-3 py-1 text-sm font-medium transition-colors',
              shouldShowSection('tool_selection')
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
            )}
            aria-pressed={shouldShowSection('tool_selection')}
          >
            Tool Seçimi
          </button>

          <button
            onClick={() => toggleFilter('category_matching')}
            className={cn(
              'rounded-full px-3 py-1 text-sm font-medium transition-colors',
              shouldShowSection('category_matching')
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
            )}
            aria-pressed={shouldShowSection('category_matching')}
          >
            Kategori Eşleştirme
          </button>

          <button
            onClick={() => toggleFilter('attention_weights')}
            className={cn(
              'rounded-full px-3 py-1 text-sm font-medium transition-colors',
              shouldShowSection('attention_weights')
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
            )}
            aria-pressed={shouldShowSection('attention_weights')}
          >
            Attention Weights
          </button>

          <button
            onClick={() => toggleFilter('thinking_steps')}
            className={cn(
              'rounded-full px-3 py-1 text-sm font-medium transition-colors',
              shouldShowSection('thinking_steps')
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
            )}
            aria-pressed={shouldShowSection('thinking_steps')}
          >
            Düşünme Adımları
          </button>

          <div className="ml-auto flex gap-2">
            <button
              onClick={showAll}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              Tümünü Göster
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={clearAll}
              className="text-sm text-gray-600 hover:text-gray-700 font-medium"
            >
              Temizle
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="space-y-6">
          {/* Tool Selection Section */}
          {shouldShowSection('tool_selection') && reasoningTrace.tool_selection && (
            <div className="animate-fadeIn">
              <ToolSelectionCard toolSelection={reasoningTrace.tool_selection} />
            </div>
          )}

          {/* Category Matching Section */}
          {shouldShowSection('category_matching') && reasoningTrace.category_matching && (
            <div className="animate-fadeIn">
              <CategoryMatchingChart categories={reasoningTrace.category_matching} />
            </div>
          )}

          {/* Attention Weights Section */}
          {shouldShowSection('attention_weights') && reasoningTrace.attention_weights && (
            <div className="animate-fadeIn">
              <AttentionWeightsChart
                attentionWeights={reasoningTrace.attention_weights}
                chartType={chartType}
                onChartTypeChange={onChartTypeChange}
              />
            </div>
          )}

          {/* Thinking Steps Section */}
          {shouldShowSection('thinking_steps') && reasoningTrace.thinking_steps && (
            <div className="animate-fadeIn">
              <ThinkingStepsTimeline steps={reasoningTrace.thinking_steps} />
            </div>
          )}

          {/* Empty state */}
          {activeFilters.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <svg
                className="h-16 w-16 text-gray-300 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                />
              </svg>
              <p className="text-gray-600 font-medium">Hiçbir filtre seçilmedi</p>
              <p className="text-sm text-gray-500 mt-1">
                Reasoning bilgilerini görmek için yukarıdan filtre seçin
              </p>
              <button
                onClick={showAll}
                className="mt-4 rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600"
              >
                Tümünü Göster
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Footer with Export */}
      <div className="border-t border-gray-200 bg-gray-50 p-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">
            {activeFilters.length} bölüm gösteriliyor
          </div>

          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <button
                className="inline-flex items-center gap-2 rounded-lg bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                aria-label="Export seçenekleri"
              >
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Export
                <svg
                  className="h-4 w-4"
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
              </button>
            </DropdownMenu.Trigger>

            <DropdownMenu.Portal>
              <DropdownMenu.Content
                className="z-50 min-w-[200px] rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
                sideOffset={5}
              >
                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
                  onSelect={() => handleExport('json')}
                >
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  JSON olarak indir
                </DropdownMenu.Item>

                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
                  onSelect={() => handleExport('pdf')}
                >
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                    />
                  </svg>
                  PDF olarak indir
                </DropdownMenu.Item>

                <DropdownMenu.Separator className="my-1 h-px bg-gray-200" />

                <DropdownMenu.Item
                  className="flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
                  onSelect={() => handleExport('share')}
                >
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
                    />
                  </svg>
                  Linki kopyala
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>
      </div>
    </div>
  );

  // Render as full-screen modal on mobile, dialog on desktop
  if (isMobile) {
    return (
      <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 animate-fadeIn" />
          <Dialog.Content
            className="fixed inset-0 z-50 bg-white animate-slideUp"
            aria-describedby="reasoning-panel-description"
          >
            <Dialog.Description id="reasoning-panel-description" className="sr-only">
              Hediye önerisi için detaylı reasoning analizi ve açıklamaları
            </Dialog.Description>
            {content}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    );
  }

  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 animate-fadeIn" />
        <Dialog.Content
          className="fixed right-0 top-0 z-50 h-full w-full max-w-3xl bg-white shadow-2xl animate-slideInRight"
          aria-describedby="reasoning-panel-description"
        >
          <Dialog.Description id="reasoning-panel-description" className="sr-only">
            Hediye önerisi için detaylı reasoning analizi ve açıklamaları
          </Dialog.Description>
          {content}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
