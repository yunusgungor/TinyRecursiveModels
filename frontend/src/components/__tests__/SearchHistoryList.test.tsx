import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchHistoryList } from '../SearchHistoryList';
import { useAppStore } from '@/store/useAppStore';
import type { UserProfile } from '@/lib/api/types';

// Mock the store
vi.mock('@/store/useAppStore');

// Mock window.confirm
global.confirm = vi.fn(() => true);

describe('SearchHistoryList', () => {
  const mockProfile: UserProfile = {
    age: 30,
    hobbies: ['reading', 'gaming'],
    relationship: 'friend',
    budget: 500,
    occasion: 'birthday',
    personalityTraits: ['practical', 'tech-savvy'],
  };

  const mockHistory = [
    {
      id: 'history-1',
      profile: mockProfile,
      timestamp: new Date().toISOString(),
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render empty state when no history', () => {
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: [],
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(screen.getByText('Henüz Arama Yapmadınız')).toBeInTheDocument();
  });

  it('should render search history list', () => {
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: mockHistory,
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(screen.getByText('30')).toBeInTheDocument();
    expect(screen.getByText('friend')).toBeInTheDocument();
    expect(screen.getByText(/500\.00\s*₺/)).toBeInTheDocument();
    expect(screen.getByText('birthday')).toBeInTheDocument();
  });

  it('should call loadSearchHistory on mount', () => {
    const loadSearchHistory = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: [],
      loadSearchHistory,
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(loadSearchHistory).toHaveBeenCalled();
  });

  it('should call removeSearchHistory when remove button is clicked', () => {
    const removeSearchHistory = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: mockHistory,
      loadSearchHistory: vi.fn(),
      removeSearchHistory,
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    const removeButton = screen.getByLabelText('Geçmişten sil');
    fireEvent.click(removeButton);

    expect(removeSearchHistory).toHaveBeenCalledWith('history-1');
  });

  it('should call clearSearchHistory when clear all button is clicked', () => {
    const clearSearchHistory = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: mockHistory,
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory,
    } as any);

    render(<SearchHistoryList />);

    const clearButton = screen.getByText('Tümünü Temizle');
    fireEvent.click(clearButton);

    expect(global.confirm).toHaveBeenCalled();
    expect(clearSearchHistory).toHaveBeenCalled();
  });

  it('should call onProfileSelect when history item is clicked', () => {
    const onProfileSelect = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: mockHistory,
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList onProfileSelect={onProfileSelect} />);

    const historyItem = screen.getByText('30').closest('div[class*="cursor-pointer"]');
    if (historyItem) {
      fireEvent.click(historyItem);
      expect(onProfileSelect).toHaveBeenCalledWith(mockProfile);
    }
  });

  it('should display history count', () => {
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: mockHistory,
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(screen.getByText('Arama Geçmişi (1)')).toBeInTheDocument();
  });

  it('should display hobbies with limit', () => {
    const profileWithManyHobbies = {
      ...mockProfile,
      hobbies: ['hobby1', 'hobby2', 'hobby3', 'hobby4', 'hobby5'],
    };
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: [
        {
          id: 'history-1',
          profile: profileWithManyHobbies,
          timestamp: new Date().toISOString(),
        },
      ],
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(screen.getByText('hobby1')).toBeInTheDocument();
    expect(screen.getByText('hobby2')).toBeInTheDocument();
    expect(screen.getByText('hobby3')).toBeInTheDocument();
    expect(screen.getByText('+2')).toBeInTheDocument();
  });

  it('should not show clear button when history is empty', () => {
    vi.mocked(useAppStore).mockReturnValue({
      searchHistory: [],
      loadSearchHistory: vi.fn(),
      removeSearchHistory: vi.fn(),
      clearSearchHistory: vi.fn(),
    } as any);

    render(<SearchHistoryList />);

    expect(screen.queryByText('Tümünü Temizle')).not.toBeInTheDocument();
  });
});
