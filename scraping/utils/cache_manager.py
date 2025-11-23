"""
Cache Manager for Gemini API Responses
Reduces API usage by caching enhancement results
"""

import json
import os
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path


class CacheManager:
    """Manages caching of Gemini API responses"""
    
    def __init__(self, cache_dir: str = "scraping/cache", ttl_days: int = 30):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.enhancement_cache_path = self.cache_dir / "enhancement_cache.json"
        self.scenario_cache_path = self.cache_dir / "scenario_cache.json"
        
        # Load caches
        self.enhancement_cache = self._load_cache(self.enhancement_cache_path)
        self.scenario_cache = self._load_cache(self.scenario_cache_path)
        
        # Statistics
        self.stats = {
            'enhancement_hits': 0,
            'enhancement_misses': 0,
            'scenario_hits': 0,
            'scenario_misses': 0
        }
        
        self.logger.info(f"CacheManager initialized with {len(self.enhancement_cache)} enhancement entries")
    
    def _load_cache(self, cache_path: Path) -> Dict[str, Any]:
        """Load cache from file"""
        if not cache_path.exists():
            return {}
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Clean expired entries
            cache = self._clean_expired_entries(cache)
            return cache
        except Exception as e:
            self.logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return {}
    
    def _save_cache(self, cache: Dict[str, Any], cache_path: Path) -> None:
        """Save cache to file"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_path}: {e}")
    
    def _clean_expired_entries(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Remove expired cache entries"""
        now = datetime.now()
        cleaned_cache = {}
        expired_count = 0
        
        for key, entry in cache.items():
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if now - timestamp < timedelta(days=self.ttl_days):
                        cleaned_cache[key] = entry
                    else:
                        expired_count += 1
                except ValueError:
                    # Invalid timestamp, keep entry
                    cleaned_cache[key] = entry
            else:
                # No timestamp, keep entry
                cleaned_cache[key] = entry
        
        if expired_count > 0:
            self.logger.info(f"Cleaned {expired_count} expired cache entries")
        
        return cleaned_cache
    
    def _generate_cache_key(self, product_name: str, price: float, category: str = "") -> str:
        """
        Generate cache key from product attributes
        
        Args:
            product_name: Product name
            price: Product price
            category: Optional category
            
        Returns:
            Cache key (hash)
        """
        # Normalize product name
        normalized_name = product_name.lower().strip()
        
        # Create key string
        key_string = f"{normalized_name}|{price:.2f}|{category}"
        
        # Generate hash
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_enhancement(self, product_name: str, price: float, category: str = "") -> Optional[Dict[str, Any]]:
        """
        Get cached enhancement for a product
        
        Args:
            product_name: Product name
            price: Product price
            category: Optional category
            
        Returns:
            Cached enhancement or None
        """
        cache_key = self._generate_cache_key(product_name, price, category)
        
        if cache_key in self.enhancement_cache:
            self.stats['enhancement_hits'] += 1
            self.logger.debug(f"Cache HIT for product: {product_name[:30]}...")
            return self.enhancement_cache[cache_key]['data']
        
        self.stats['enhancement_misses'] += 1
        return None
    
    def set_enhancement(self, product_name: str, price: float, enhancement: Dict[str, Any], category: str = "") -> None:
        """
        Cache enhancement for a product
        
        Args:
            product_name: Product name
            price: Product price
            enhancement: Enhancement data
            category: Optional category
        """
        cache_key = self._generate_cache_key(product_name, price, category)
        
        self.enhancement_cache[cache_key] = {
            'data': enhancement,
            'timestamp': datetime.now().isoformat(),
            'product_name': product_name,
            'price': price
        }
        
        # Save to disk periodically (every 10 entries)
        if len(self.enhancement_cache) % 10 == 0:
            self._save_cache(self.enhancement_cache, self.enhancement_cache_path)
    
    def get_scenario(self, profile_signature: str) -> Optional[Dict[str, Any]]:
        """
        Get cached scenario for a profile signature
        
        Args:
            profile_signature: Unique signature of the profile
            
        Returns:
            Cached scenario or None
        """
        if profile_signature in self.scenario_cache:
            self.stats['scenario_hits'] += 1
            return self.scenario_cache[profile_signature]['data']
        
        self.stats['scenario_misses'] += 1
        return None
    
    def set_scenario(self, profile_signature: str, scenario: Dict[str, Any]) -> None:
        """
        Cache scenario for a profile signature
        
        Args:
            profile_signature: Unique signature of the profile
            scenario: Scenario data
        """
        self.scenario_cache[profile_signature] = {
            'data': scenario,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to disk periodically
        if len(self.scenario_cache) % 10 == 0:
            self._save_cache(self.scenario_cache, self.scenario_cache_path)
    
    def save_all(self) -> None:
        """Save all caches to disk"""
        self._save_cache(self.enhancement_cache, self.enhancement_cache_path)
        self._save_cache(self.scenario_cache, self.scenario_cache_path)
        self.logger.info("All caches saved to disk")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_enhancement = self.stats['enhancement_hits'] + self.stats['enhancement_misses']
        total_scenario = self.stats['scenario_hits'] + self.stats['scenario_misses']
        
        enhancement_hit_rate = (
            self.stats['enhancement_hits'] / total_enhancement * 100 
            if total_enhancement > 0 else 0
        )
        scenario_hit_rate = (
            self.stats['scenario_hits'] / total_scenario * 100 
            if total_scenario > 0 else 0
        )
        
        return {
            'enhancement_cache_size': len(self.enhancement_cache),
            'scenario_cache_size': len(self.scenario_cache),
            'enhancement_hits': self.stats['enhancement_hits'],
            'enhancement_misses': self.stats['enhancement_misses'],
            'enhancement_hit_rate': f"{enhancement_hit_rate:.1f}%",
            'scenario_hits': self.stats['scenario_hits'],
            'scenario_misses': self.stats['scenario_misses'],
            'scenario_hit_rate': f"{scenario_hit_rate:.1f}%",
            'api_calls_saved': self.stats['enhancement_hits'] + self.stats['scenario_hits']
        }
    
    def clear_cache(self, cache_type: str = 'all') -> None:
        """
        Clear cache
        
        Args:
            cache_type: 'enhancement', 'scenario', or 'all'
        """
        if cache_type in ['enhancement', 'all']:
            self.enhancement_cache = {}
            self._save_cache(self.enhancement_cache, self.enhancement_cache_path)
            self.logger.info("Enhancement cache cleared")
        
        if cache_type in ['scenario', 'all']:
            self.scenario_cache = {}
            self._save_cache(self.scenario_cache, self.scenario_cache_path)
            self.logger.info("Scenario cache cleared")
