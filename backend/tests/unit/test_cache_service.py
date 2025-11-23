"""Unit tests for cache service"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.cache_service import CacheService
from app.models.schemas import UserProfile, GiftItem, GiftRecommendation
from app.core.exceptions import CacheError


@pytest.fixture
def mock_redis():
    """Create mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock()
    redis_mock.delete = AsyncMock()
    redis_mock.ttl = AsyncMock(return_value=3600)
    redis_mock.info = AsyncMock(return_value={'used_memory': 1024 * 1024})  # 1 MB
    redis_mock.scan_iter = AsyncMock()
    redis_mock.close = AsyncMock()
    return redis_mock


@pytest.fixture
def cache_service(mock_redis):
    """Create cache service with mock Redis"""
    service = CacheService(redis_client=mock_redis)
    return service


@pytest.fixture
def sample_profile():
    """Sample user profile"""
    return UserProfile(
        age=35,
        hobbies=["gardening", "cooking"],
        relationship="mother",
        budget=500.0,
        occasion="birthday",
        personality_traits=["practical", "eco-friendly"]
    )


@pytest.fixture
def sample_recommendations(sample_profile):
    """Sample gift recommendations"""
    gift = GiftItem(
        id="12345",
        name="Premium Coffee Set",
        category="Kitchen & Dining",
        price=299.99,
        rating=4.5,
        image_url="https://cdn.trendyol.com/example.jpg",
        trendyol_url="https://www.trendyol.com/product/12345",
        description="High-quality coffee set",
        tags=["coffee", "kitchen", "gift"],
        age_suitability=(25, 65),
        occasion_fit=["birthday", "anniversary"],
        in_stock=True
    )
    
    return [
        GiftRecommendation(
            gift=gift,
            confidence_score=0.85,
            reasoning=["Category match", "Price within budget"],
            tool_insights={},
            rank=1
        )
    ]


class TestCacheServiceConnection:
    """Test cache service connection"""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis):
        """Test successful Redis connection"""
        service = CacheService(redis_client=mock_redis)
        await service.connect()
        
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Redis connection failure"""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))
        
        service = CacheService(redis_client=mock_redis)
        
        with pytest.raises(CacheError):
            await service.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, cache_service, mock_redis):
        """Test Redis disconnection"""
        await cache_service.disconnect()
        
        mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, cache_service, mock_redis):
        """Test successful health check"""
        result = await cache_service.health_check()
        
        assert result is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, cache_service, mock_redis):
        """Test failed health check"""
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection lost"))
        
        result = await cache_service.health_check()
        
        assert result is False


class TestProfileHashing:
    """Test profile hash generation"""
    
    def test_generate_profile_hash_consistency(self, cache_service, sample_profile):
        """Test that same profile generates same hash"""
        hash1 = cache_service._generate_profile_hash(sample_profile)
        hash2 = cache_service._generate_profile_hash(sample_profile)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
    
    def test_generate_profile_hash_different_profiles(self, cache_service):
        """Test that different profiles generate different hashes"""
        profile1 = UserProfile(
            age=35,
            hobbies=["gardening"],
            relationship="mother",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        profile2 = UserProfile(
            age=40,
            hobbies=["cooking"],
            relationship="father",
            budget=1000.0,
            occasion="anniversary",
            personality_traits=["adventurous"]
        )
        
        hash1 = cache_service._generate_profile_hash(profile1)
        hash2 = cache_service._generate_profile_hash(profile2)
        
        assert hash1 != hash2
    
    def test_generate_profile_hash_order_independence(self, cache_service):
        """Test that hobby order doesn't affect hash"""
        profile1 = UserProfile(
            age=35,
            hobbies=["gardening", "cooking"],
            relationship="mother",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical", "eco-friendly"]
        )
        
        profile2 = UserProfile(
            age=35,
            hobbies=["cooking", "gardening"],  # Different order
            relationship="mother",
            budget=500.0,
            occasion="birthday",
            personality_traits=["eco-friendly", "practical"]  # Different order
        )
        
        hash1 = cache_service._generate_profile_hash(profile1)
        hash2 = cache_service._generate_profile_hash(profile2)
        
        assert hash1 == hash2


class TestCacheGetSet:
    """Test cache get/set operations"""
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_service, sample_profile, mock_redis):
        """Test cache miss returns None"""
        mock_redis.get = AsyncMock(return_value=None)
        
        result = await cache_service.get_recommendations(sample_profile)
        
        assert result is None
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_hit(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test cache hit returns recommendations"""
        # Prepare cached data
        cached_data = json.dumps([rec.model_dump(mode='json') for rec in sample_recommendations])
        mock_redis.get = AsyncMock(return_value=cached_data)
        
        result = await cache_service.get_recommendations(sample_profile)
        
        assert result is not None
        assert len(result) == 1
        assert result[0].gift.id == "12345"
        assert result[0].confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_set_recommendations(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test setting recommendations in cache"""
        # Mock cache size check
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations,
            ttl=3600
        )
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        # Verify key format
        assert call_args[0][0].startswith("recommendations:")
        # Verify TTL
        assert call_args[0][1] == 3600
        # Verify data is JSON
        json.loads(call_args[0][2])  # Should not raise
    
    @pytest.mark.asyncio
    async def test_set_recommendations_without_redis(
        self,
        sample_profile,
        sample_recommendations
    ):
        """Test setting recommendations without Redis connection"""
        service = CacheService(redis_client=None)
        
        # Should not raise, just log warning
        await service.set_recommendations(sample_profile, sample_recommendations)
    
    @pytest.mark.asyncio
    async def test_get_recommendations_without_redis(self, sample_profile):
        """Test getting recommendations without Redis connection"""
        service = CacheService(redis_client=None)
        
        result = await service.get_recommendations(sample_profile)
        
        assert result is None


class TestTTLManagement:
    """Test TTL (Time To Live) management"""
    
    @pytest.mark.asyncio
    async def test_default_ttl_for_recommendations(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test default TTL is used for recommendations"""
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations
        )
        
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == cache_service.default_ttl
    
    @pytest.mark.asyncio
    async def test_custom_ttl_for_recommendations(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test custom TTL can be set"""
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        custom_ttl = 7200
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations,
            ttl=custom_ttl
        )
        
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == custom_ttl
    
    @pytest.mark.asyncio
    async def test_trendyol_data_ttl(self, cache_service, mock_redis):
        """Test Trendyol data uses correct TTL"""
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        
        await cache_service.set_trendyol_data(
            "test_key",
            {"data": "test"}
        )
        
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == cache_service.trendyol_ttl


class TestCacheEviction:
    """Test cache eviction policy"""
    
    @pytest.mark.asyncio
    async def test_eviction_when_size_exceeded(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test eviction occurs when cache size is exceeded"""
        # Mock large cache size
        large_size = cache_service.max_size_bytes + 1000
        mock_redis.info = AsyncMock(return_value={'used_memory': large_size})
        
        # Mock scan_iter to return some keys
        async def mock_scan_iter(match=None):
            keys = ["recommendations:hash1", "recommendations:hash2"]
            for key in keys:
                yield key
        
        mock_redis.scan_iter = mock_scan_iter
        mock_redis.ttl = AsyncMock(return_value=1800)
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations
        )
        
        # Verify delete was called for eviction
        assert mock_redis.delete.call_count > 0
    
    @pytest.mark.asyncio
    async def test_no_eviction_when_size_ok(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test no eviction when cache size is within limits"""
        # Mock small cache size
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations
        )
        
        # Verify delete was not called
        mock_redis.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_eviction_prioritizes_shortest_ttl(
        self,
        cache_service,
        sample_profile,
        sample_recommendations,
        mock_redis
    ):
        """Test eviction removes entries with shortest TTL first"""
        # Mock large cache size
        large_size = cache_service.max_size_bytes + 1000
        mock_redis.info = AsyncMock(return_value={'used_memory': large_size})
        
        # Mock keys with different TTLs
        keys_with_ttl = [
            ("recommendations:hash1", 3600),
            ("recommendations:hash2", 1800),
            ("recommendations:hash3", 900),
        ]
        
        async def mock_scan_iter(match=None):
            for key, _ in keys_with_ttl:
                yield key
        
        ttl_map = {key: ttl for key, ttl in keys_with_ttl}
        
        async def mock_ttl(key):
            return ttl_map.get(key, -1)
        
        mock_redis.scan_iter = mock_scan_iter
        mock_redis.ttl = mock_ttl
        
        deleted_keys = []
        
        async def mock_delete(key):
            deleted_keys.append(key)
        
        mock_redis.delete = mock_delete
        
        await cache_service.set_recommendations(
            sample_profile,
            sample_recommendations
        )
        
        # Verify shortest TTL was deleted first
        if deleted_keys:
            assert deleted_keys[0] == "recommendations:hash3"


class TestCacheSizeMonitoring:
    """Test cache size monitoring"""
    
    @pytest.mark.asyncio
    async def test_get_cache_size(self, cache_service, mock_redis):
        """Test getting cache size"""
        expected_size = 5 * 1024 * 1024  # 5 MB
        mock_redis.info = AsyncMock(return_value={'used_memory': expected_size})
        
        size = await cache_service.get_cache_size()
        
        assert size == expected_size
    
    @pytest.mark.asyncio
    async def test_get_cache_size_without_redis(self):
        """Test getting cache size without Redis"""
        service = CacheService(redis_client=None)
        
        size = await service.get_cache_size()
        
        assert size == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_service, mock_redis):
        """Test getting cache statistics"""
        mock_redis.info = AsyncMock(return_value={'used_memory': 10 * 1024 * 1024})
        
        # Mock scan_iter for counting keys
        async def mock_scan_iter(match=None):
            if "recommendations:" in match:
                for i in range(5):
                    yield f"recommendations:hash{i}"
            elif "trendyol:" in match:
                for i in range(3):
                    yield f"trendyol:key{i}"
        
        mock_redis.scan_iter = mock_scan_iter
        
        stats = await cache_service.get_cache_stats()
        
        assert stats["connected"] is True
        assert stats["size_mb"] == 10.0
        assert stats["total_keys"] == 8
        assert stats["recommendation_keys"] == 5
        assert stats["trendyol_keys"] == 3
        assert "utilization_percent" in stats
    
    @pytest.mark.asyncio
    async def test_get_cache_stats_without_redis(self):
        """Test getting cache stats without Redis"""
        service = CacheService(redis_client=None)
        
        stats = await service.get_cache_stats()
        
        assert stats["connected"] is False
        assert stats["size_bytes"] == 0


class TestTrendyolDataCaching:
    """Test Trendyol data caching"""
    
    @pytest.mark.asyncio
    async def test_get_trendyol_data_miss(self, cache_service, mock_redis):
        """Test Trendyol data cache miss"""
        mock_redis.get = AsyncMock(return_value=None)
        
        result = await cache_service.get_trendyol_data("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_trendyol_data_hit(self, cache_service, mock_redis):
        """Test Trendyol data cache hit"""
        test_data = {"products": [{"id": "123", "name": "Test"}]}
        mock_redis.get = AsyncMock(return_value=json.dumps(test_data))
        
        result = await cache_service.get_trendyol_data("test_key")
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_set_trendyol_data(self, cache_service, mock_redis):
        """Test setting Trendyol data in cache"""
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024})
        test_data = {"products": [{"id": "123", "name": "Test"}]}
        
        await cache_service.set_trendyol_data("test_key", test_data)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        # Verify key format
        assert call_args[0][0] == "trendyol:test_key"
        # Verify data
        assert json.loads(call_args[0][2]) == test_data


class TestCacheClear:
    """Test cache clearing"""
    
    @pytest.mark.asyncio
    async def test_clear_all_cache(self, cache_service, mock_redis):
        """Test clearing all cache entries"""
        async def mock_scan_iter(match=None):
            keys = ["key1", "key2", "key3"]
            for key in keys:
                yield key
        
        mock_redis.scan_iter = mock_scan_iter
        
        deleted_count = await cache_service.clear_cache()
        
        assert deleted_count == 3
        assert mock_redis.delete.call_count == 3
    
    @pytest.mark.asyncio
    async def test_clear_cache_with_pattern(self, cache_service, mock_redis):
        """Test clearing cache with pattern"""
        async def mock_scan_iter(match=None):
            if "recommendations:" in match:
                keys = ["recommendations:hash1", "recommendations:hash2"]
                for key in keys:
                    yield key
        
        mock_redis.scan_iter = mock_scan_iter
        
        deleted_count = await cache_service.clear_cache("recommendations:*")
        
        assert deleted_count == 2
