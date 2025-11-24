import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { GiftItem, UserProfile } from '@/lib/api/types';
import {
  getFavorites,
  addFavorite as addFavoriteToStorage,
  removeFavorite as removeFavoriteFromStorage,
  isFavorite as checkIsFavorite,
  clearFavorites as clearFavoritesFromStorage,
  getSearchHistory,
  addSearchHistory as addSearchHistoryToStorage,
  removeSearchHistory as removeSearchHistoryFromStorage,
  clearSearchHistory as clearSearchHistoryFromStorage,
  type FavoriteItem,
  type SearchHistoryItem,
} from '@/lib/utils/localStorage';

interface AppState {
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
  toggleTheme: () => void;
  
  // Favorites
  favorites: FavoriteItem[];
  loadFavorites: () => void;
  addFavorite: (gift: GiftItem) => void;
  removeFavorite: (giftId: string) => void;
  isFavorite: (giftId: string) => boolean;
  clearFavorites: () => void;
  
  // Search History
  searchHistory: SearchHistoryItem[];
  loadSearchHistory: () => void;
  addSearchHistory: (profile: UserProfile) => void;
  removeSearchHistory: (id: string) => void;
  clearSearchHistory: () => void;
  
  // Comparison Mode
  selectedGiftsForComparison: string[];
  isComparisonMode: boolean;
  toggleGiftSelection: (giftId: string) => void;
  clearGiftSelection: () => void;
  setComparisonMode: (enabled: boolean) => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'light',
        setTheme: (theme) => set({ theme }),
        toggleTheme: () =>
          set((state) => ({
            theme: state.theme === 'light' ? 'dark' : 'light',
          })),
        
        // Favorites
        favorites: [],
        loadFavorites: () => {
          const favorites = getFavorites();
          set({ favorites });
        },
        addFavorite: (gift) => {
          addFavoriteToStorage(gift);
          const favorites = getFavorites();
          set({ favorites });
        },
        removeFavorite: (giftId) => {
          removeFavoriteFromStorage(giftId);
          const favorites = getFavorites();
          set({ favorites });
        },
        isFavorite: (giftId) => {
          return checkIsFavorite(giftId);
        },
        clearFavorites: () => {
          clearFavoritesFromStorage();
          set({ favorites: [] });
        },
        
        // Search History
        searchHistory: [],
        loadSearchHistory: () => {
          const searchHistory = getSearchHistory();
          set({ searchHistory });
        },
        addSearchHistory: (profile) => {
          addSearchHistoryToStorage(profile);
          const searchHistory = getSearchHistory();
          set({ searchHistory });
        },
        removeSearchHistory: (id) => {
          removeSearchHistoryFromStorage(id);
          const searchHistory = getSearchHistory();
          set({ searchHistory });
        },
        clearSearchHistory: () => {
          clearSearchHistoryFromStorage();
          set({ searchHistory: [] });
        },
        
        // Comparison Mode
        selectedGiftsForComparison: [],
        isComparisonMode: false,
        toggleGiftSelection: (giftId) => {
          set((state) => {
            const isSelected = state.selectedGiftsForComparison.includes(giftId);
            return {
              selectedGiftsForComparison: isSelected
                ? state.selectedGiftsForComparison.filter((id) => id !== giftId)
                : [...state.selectedGiftsForComparison, giftId],
            };
          });
        },
        clearGiftSelection: () => {
          set({ selectedGiftsForComparison: [], isComparisonMode: false });
        },
        setComparisonMode: (enabled) => {
          set({ isComparisonMode: enabled });
          if (!enabled) {
            set({ selectedGiftsForComparison: [] });
          }
        },
      }),
      {
        name: 'app-storage',
      }
    )
  )
);
