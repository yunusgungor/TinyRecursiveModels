"""Cache service for Redis-based caching"""

import json
import hashlib
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis

from app.models.schemas import UserProfile, GiftRecommendation
from app.core.config import settings
from app.core.exceptions import CacheError


logger = logging.getLogger(__name__)


class CacheService:
    """Service for Redis-based caching with TTL and eviction policies"""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initialize cache service
        
        Args:
            redis_client: Redis client instance (optional)
        """
        self.redis = redis_client
        self.default_ttl = settings.CACHE_TTL_RECOMMENDATIONS
        self.trendyol_ttl = settings.CACHE_TTL_TRENDYOL_DATA
        self.max_size_bytes = settings.CACHE_MAX_SIZE_MB * 1024 * 1024
        
        logger.info("CacheService initialized")
    
    async def connect(self) -> None:
        """
        Connect to Redis server
        
        Raises:
            CacheError: If connection fails
        """
        try:
            if self.redis is None:
                self.redis = await redis.from_url(
                    f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    encoding="utf-8",
                    decode_responses=True
                )
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise CacheError(f"Failed to connect to Redis: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")
    
    def _generate_profile_hash(self, profile: UserProfile) -> str:
        """
        Generate unique hash for user profile
        
        Args:
            profile: User profile data
            
        Returns:
            str: SHA256 hash of profile
        """
        # Create a deterministic string representation
        profile_str = json.dumps(
            {
                "age": profile.age,
                "hobbies": sorted(profile.hobbies),  # Sort for consistency
                "relationship": profile.relationship,
                "budget": profile.budget,
                "occasion": profile.occasion,
                "personality_traits": sorted(profile.personality_traits),
            },
            sort_keys=True
        )
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(profile_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    async def get_recommendations(
        self,
        profile: UserProfile
    ) -> Optional[List[GiftRecommendation]]:
        """
        Get cached recommendations for user profile
        
        Args:
            profile: User profile data
            
        Returns:
            List of recommendations if found, None otherwise
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache get")
            return None
        
        try:
            profile_hash = self._generate_profile_hash(profile)
            cache_key = f"recommendations:{profile_hash}"
            
            # Get from cache
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache hit for profile hash: {profile_hash}")
                
                # Parse JSON data
                data = json.loads(cached_data)
                
                # Convert to GiftRecommendation objects
                recommendations = [
                    GiftRecommendation(**rec) for rec in data
                ]
                
                return recommendations
            else:
                logger.info(f"Cache miss for profile hash: {profile_hash}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get from cache: {str(e)}")
            return None
    
    async def set_recommendations(
        self,
        profile: UserProfile,
        recommendations: List[GiftRecommendation],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache recommendations for user profile
        
        Args:
            profile: User profile data
            recommendations: List of recommendations to cache
            ttl: Time to live in seconds (default: from settings)
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache set")
            return
        
        try:
            profile_hash = self._generate_profile_hash(profile)
            cache_key = f"recommendations:{profile_hash}"
            ttl = ttl or self.default_ttl
            
            # Convert recommendations to JSON-serializable format
            data = [rec.model_dump(mode='json') for rec in recommendations]
            json_data = json.dumps(data)
            
            # Check cache size before setting
            await self._check_and_evict_if_needed(len(json_data))
            
            # Set in cache with TTL
            await self.redis.setex(
                cache_key,
                ttl,
                json_data
            )
            
            logger.info(
                f"Cached recommendations for profile hash: {profile_hash} "
                f"with TTL: {ttl}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to set cache: {str(e)}")
    
    async def get_trendyol_data(
        self,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached Trendyol API data
        
        Args:
            cache_key: Cache key for Trendyol data
            
        Returns:
            Cached data if found, None otherwise
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache get")
            return None
        
        try:
            full_key = f"trendyol:{cache_key}"
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                logger.info(f"Cache hit for Trendyol data: {cache_key}")
                return json.loads(cached_data)
            else:
                logger.info(f"Cache miss for Trendyol data: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get Trendyol data from cache: {str(e)}")
            return None
    
    async def set_trendyol_data(
        self,
        cache_key: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache Trendyol API data
        
        Args:
            cache_key: Cache key for Trendyol data
            data: Data to cache
            ttl: Time to live in seconds (default: 30 minutes)
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache set")
            return
        
        try:
            full_key = f"trendyol:{cache_key}"
            ttl = ttl or self.trendyol_ttl
            
            json_data = json.dumps(data)
            
            # Check cache size before setting
            await self._check_and_evict_if_needed(len(json_data))
            
            # Set in cache with TTL
            await self.redis.setex(
                full_key,
                ttl,
                json_data
            )
            
            logger.info(
                f"Cached Trendyol data: {cache_key} with TTL: {ttl}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to set Trendyol data cache: {str(e)}")
    
    async def get_cache_size(self) -> int:
        """
        Get current cache size in bytes
        
        Returns:
            Cache size in bytes
        """
        if not self.redis:
            return 0
        
        try:
            info = await self.redis.info('memory')
            used_memory = info.get('used_memory', 0)
            return used_memory
            
        except Exception as e:
            logger.error(f"Failed to get cache size: {str(e)}")
            return 0
    
    async def _check_and_evict_if_needed(self, new_data_size: int) -> None:
        """
        Check cache size and evict oldest entries if needed
        
        Args:
            new_data_size: Size of new data to be added
        """
        try:
            current_size = await self.get_cache_size()
            
            # Check if adding new data would exceed max size
            if current_size + new_data_size > self.max_size_bytes:
                logger.warning(
                    f"Cache size ({current_size} bytes) approaching limit "
                    f"({self.max_size_bytes} bytes), evicting oldest entries"
                )
                
                # Get all keys with TTL
                keys_to_evict = []
                
                # Scan for recommendation keys
                async for key in self.redis.scan_iter(match="recommendations:*"):
                    ttl = await self.redis.ttl(key)
                    if ttl > 0:
                        keys_to_evict.append((key, ttl))
                
                # Scan for Trendyol keys
                async for key in self.redis.scan_iter(match="trendyol:*"):
                    ttl = await self.redis.ttl(key)
                    if ttl > 0:
                        keys_to_evict.append((key, ttl))
                
                # Sort by TTL (evict entries with shortest TTL first)
                keys_to_evict.sort(key=lambda x: x[1])
                
                # Evict oldest entries until we have enough space
                evicted_count = 0
                for key, _ in keys_to_evict:
                    await self.redis.delete(key)
                    evicted_count += 1
                    
                    # Check if we have enough space now
                    current_size = await self.get_cache_size()
                    if current_size + new_data_size <= self.max_size_bytes:
                        break
                
                logger.info(f"Evicted {evicted_count} cache entries")
                
        except Exception as e:
            logger.error(f"Failed to check and evict cache: {str(e)}")
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching pattern
        
        Args:
            pattern: Pattern to match keys (default: all keys)
            
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            logger.warning("Redis not connected, skipping cache clear")
            return 0
        
        try:
            pattern = pattern or "*"
            deleted_count = 0
            
            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key)
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} cache entries matching: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.redis:
            return {
                "connected": False,
                "size_bytes": 0,
                "size_mb": 0.0,
                "max_size_mb": settings.CACHE_MAX_SIZE_MB,
                "utilization_percent": 0.0,
                "total_keys": 0,
                "recommendation_keys": 0,
                "trendyol_keys": 0,
            }
        
        try:
            # Get memory info
            info = await self.redis.info('memory')
            used_memory = info.get('used_memory', 0)
            
            # Count keys by type
            recommendation_keys = 0
            trendyol_keys = 0
            
            async for key in self.redis.scan_iter(match="recommendations:*"):
                recommendation_keys += 1
            
            async for key in self.redis.scan_iter(match="trendyol:*"):
                trendyol_keys += 1
            
            total_keys = recommendation_keys + trendyol_keys
            
            size_mb = used_memory / (1024 * 1024)
            utilization = (used_memory / self.max_size_bytes) * 100
            
            return {
                "connected": True,
                "size_bytes": used_memory,
                "size_mb": round(size_mb, 2),
                "max_size_mb": settings.CACHE_MAX_SIZE_MB,
                "utilization_percent": round(utilization, 2),
                "total_keys": total_keys,
                "recommendation_keys": recommendation_keys,
                "trendyol_keys": trendyol_keys,
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """
        Check if Redis is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False


# Singleton instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get or create cache service singleton"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    
    return _cache_service
