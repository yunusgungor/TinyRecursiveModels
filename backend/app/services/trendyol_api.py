"""
Trendyol API Service - Scraping Implementation
This module wraps the scraping service and maintains backward compatibility
"""

# Export everything from scraping service for backward compatibility
from app.services.trendyol_scraping_service import (
    TrendyolProduct,
    TrendyolScrapingService as TrendyolAPIService,
    get_trendyol_scraping_service as get_trendyol_service,
    cleanup_trendyol_service
)

__all__ = [
    'TrendyolProduct',
    'TrendyolAPIService',
    'get_trendyol_service',
    'cleanup_trendyol_service'
]
