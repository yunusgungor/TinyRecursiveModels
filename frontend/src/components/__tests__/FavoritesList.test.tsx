import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FavoritesList } from '../FavoritesList';
import { useAppStore } from '@/store/useAppStore';
import type { GiftItem } from '@/lib/api/types';

// Mock the store
vi.mock('@/store/useAppStore');

describe('FavoritesList', () => {
  const mockGift: GiftItem = {
    id: 'test-1',
    name: 'Test Gift',
    category: 'Electronics',
    price: 100,
    rating: 4.5,
    imageUrl: 'https://example.com/image.jpg',
    trendyolUrl: 'https://trendyol.com/product/test-1',
    description: 'Test description',
    tags: ['test', 'gift'],
    ageSuitability: [18, 65],
    occasionFit: ['birthday'],
    inStock: true,
  };

  const mockFavorites = [
    {
      gift: mockGift,
      addedAt: new Date().toISOString(),
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render empty state when no favorites', () => {
    vi.mocked(useAppStore).mockReturnValue({
      favorites: [],
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList />);

    expect(screen.getByText('Henüz Favori Eklemediniz')).toBeInTheDocument();
  });

  it('should render favorites list', () => {
    vi.mocked(useAppStore).mockReturnValue({
      favorites: mockFavorites,
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList />);

    expect(screen.getByText('Test Gift')).toBeInTheDocument();
    expect(screen.getByText('Electronics')).toBeInTheDocument();
    expect(screen.getByText(/100\.00\s*₺/)).toBeInTheDocument();
  });

  it('should call loadFavorites on mount', () => {
    const loadFavorites = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      favorites: [],
      loadFavorites,
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList />);

    expect(loadFavorites).toHaveBeenCalled();
  });

  it('should call removeFavorite when remove button is clicked', () => {
    const removeFavorite = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      favorites: mockFavorites,
      loadFavorites: vi.fn(),
      removeFavorite,
    } as any);

    render(<FavoritesList />);

    const removeButton = screen.getByLabelText('Favorilerden çıkar');
    fireEvent.click(removeButton);

    expect(removeFavorite).toHaveBeenCalledWith('test-1');
  });

  it('should call onGiftClick when card is clicked', () => {
    const onGiftClick = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      favorites: mockFavorites,
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList onGiftClick={onGiftClick} />);

    const card = screen.getByText('Test Gift').closest('div[class*="cursor-pointer"]');
    if (card) {
      fireEvent.click(card);
      expect(onGiftClick).toHaveBeenCalledWith(mockGift);
    }
  });

  it('should call onTrendyolClick when Trendyol button is clicked', () => {
    const onTrendyolClick = vi.fn();
    vi.mocked(useAppStore).mockReturnValue({
      favorites: mockFavorites,
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList onTrendyolClick={onTrendyolClick} />);

    const trendyolButton = screen.getByText("Trendyol'da Gör");
    fireEvent.click(trendyolButton);

    expect(onTrendyolClick).toHaveBeenCalledWith(mockGift.trendyolUrl);
  });

  it('should show out of stock badge for unavailable items', () => {
    const outOfStockGift = { ...mockGift, inStock: false };
    vi.mocked(useAppStore).mockReturnValue({
      favorites: [{ gift: outOfStockGift, addedAt: new Date().toISOString() }],
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList />);

    expect(screen.getByText('Stokta Yok')).toBeInTheDocument();
  });

  it('should display favorites count', () => {
    vi.mocked(useAppStore).mockReturnValue({
      favorites: mockFavorites,
      loadFavorites: vi.fn(),
      removeFavorite: vi.fn(),
    } as any);

    render(<FavoritesList />);

    expect(screen.getByText('Favorilerim (1)')).toBeInTheDocument();
  });
});
