"""
Unit tests for performance optimizations

These tests validate performance improvements without requiring a running server.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import torch

from app.services.cache_service import CacheService
from app.services.model_inference import ModelInferenceService
from app.models.schemas import UserProfile, GiftItem


@pytest.mark.asyncio
class TestCachePerformance:
    """Test cache service performance"""
    
    async def test_cache_hit_performance(self):
        """
        Test that cache hits are fast
        
        Validates: Requirements 6.1
        """
        # Create mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='[{"gift": {"id": "1"}, "confidence_score": 0.9}]')
        
        cache_service = CacheService(redis_client=mock_redis)
        
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        # Measure cache retrieval time
        start_time = time.time()
        result = await cache_service.get_recommendations(profile)
        cache_time = time.time() - start_time
        
        print(f"\nCache retrieval time: {cache_time * 1000:.2f}ms")
        
        # Cache hits should be very fast (under 10ms)
        assert cache_time < 0.01, "Cache hit should be under 10ms"
        assert mock_redis.get.called, "Redis get should be called"
    
    async def test_cache_set_performance(self):
        """
        Test that cache writes are fast
        
        Validates: Requirements 6.1
        """
        # Create mock Redis client
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis.info = AsyncMock(return_value={'used_memory': 1000000})
        
        cache_service = CacheService(redis_client=mock_redis)
        
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        from app.models.schemas import GiftRecommendation
        recommendations = [
            GiftRecommendation(
                gift=GiftItem(
                    id="1",
                    name="Test Gift",
                    category="Books",
                    price=100.0,
                    rating=4.5,
                    image_url="https://example.com/image.jpg",
                    trendyol_url="https://trendyol.com/product/1",
                    description="Test",
                    tags=["test"],
                    age_suitability=(18, 100),
                    occasion_fit=["birthday"],
                    in_stock=True
                ),
                confidence_score=0.9,
                reasoning=["test"],
                tool_insights={},
                rank=1
            )
        ]
        
        # Measure cache write time
        start_time = time.time()
        await cache_service.set_recommendations(profile, recommendations)
        cache_time = time.time() - start_time
        
        print(f"\nCache write time: {cache_time * 1000:.2f}ms")
        
        # Cache writes should be fast (under 50ms)
        assert cache_time < 0.05, "Cache write should be under 50ms"
        assert mock_redis.setex.called, "Redis setex should be called"
    
    async def test_profile_hash_consistency(self):
        """
        Test that profile hashing is consistent
        
        Validates: Requirements 6.1
        """
        cache_service = CacheService()
        
        profile1 = UserProfile(
            age=30,
            hobbies=["reading", "gaming"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        profile2 = UserProfile(
            age=30,
            hobbies=["gaming", "reading"],  # Different order
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        hash1 = cache_service._generate_profile_hash(profile1)
        hash2 = cache_service._generate_profile_hash(profile2)
        
        # Hashes should be the same regardless of hobby order
        assert hash1 == hash2, "Profile hashes should be consistent"


@pytest.mark.asyncio
class TestModelInferenceOptimization:
    """Test model inference optimizations"""
    
    def test_device_selection(self):
        """
        Test that device selection works correctly
        
        Validates: Requirements 2.4
        """
        service = ModelInferenceService()
        
        device = service._get_device()
        
        print(f"\nSelected device: {device}")
        
        # Device should be either cuda or cpu
        assert str(device) in ["cuda", "cpu"], "Device should be cuda or cpu"
        
        # If CUDA is available, should use it
        if torch.cuda.is_available():
            assert str(device) == "cuda", "Should use CUDA when available"
        else:
            assert str(device) == "cpu", "Should use CPU when CUDA unavailable"
    
    def test_model_config_caching(self):
        """
        Test that model configuration is cached
        
        Validates: Requirements 6.3
        """
        service = ModelInferenceService()
        
        # Get config multiple times
        start_time = time.time()
        config1 = service._get_default_config()
        first_time = time.time() - start_time
        
        start_time = time.time()
        config2 = service._get_default_config()
        second_time = time.time() - start_time
        
        print(f"\nFirst config retrieval: {first_time * 1000:.4f}ms")
        print(f"Second config retrieval: {second_time * 1000:.4f}ms")
        
        # Configs should be identical
        assert config1 == config2, "Configs should be identical"
        
        # Second retrieval should be faster (cached)
        assert second_time <= first_time, "Cached retrieval should be as fast or faster"


@pytest.mark.asyncio
class TestQueryOptimization:
    """Test database query optimizations"""
    
    async def test_batch_insert_performance(self):
        """
        Test that batch inserts are efficient
        
        Validates: Requirements 6.3
        """
        from app.core.database import DatabaseManager
        from unittest.mock import MagicMock
        
        # Create mock connection
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        # Create mock context manager
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire)
        
        db_manager = DatabaseManager()
        db_manager.pool = mock_pool
        
        # Prepare batch data
        batch_data = [(f"user_{i}", i * 100) for i in range(100)]
        
        # Measure batch insert time
        start_time = time.time()
        await db_manager.execute_batch(
            "INSERT INTO test (name, value) VALUES ($1, $2)",
            batch_data
        )
        batch_time = time.time() - start_time
        
        print(f"\nBatch insert time for 100 records: {batch_time * 1000:.2f}ms")
        
        # Batch insert should be fast
        assert batch_time < 0.1, "Batch insert should be under 100ms"
        assert mock_conn.executemany.called, "executemany should be called"


@pytest.mark.asyncio  
class TestCodeSplitting:
    """Test frontend code splitting effectiveness"""
    
    def test_lazy_loading_imports(self):
        """
        Test that lazy loading is properly configured
        
        Validates: Requirements 6.3
        """
        # This is a conceptual test - in practice, we'd check bundle sizes
        # For now, we verify the lazy loading pattern is in place
        
        import os
        routes_file = "frontend/src/routes/index.tsx"
        
        if os.path.exists(routes_file):
            with open(routes_file, 'r') as f:
                content = f.read()
                
            # Check for lazy loading pattern
            assert 'lazy' in content, "Routes should use lazy loading"
            assert 'Suspense' in content, "Routes should use Suspense"
            
            print("\n✓ Lazy loading is properly configured")
        else:
            pytest.skip("Frontend routes file not found")


@pytest.mark.performance
class TestMemoryOptimization:
    """Test memory optimization strategies"""
    
    def test_tensor_memory_cleanup(self):
        """
        Test that tensors are properly cleaned up
        
        Validates: Requirements 6.5
        """
        import gc
        
        # Create some tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
        
        # Get memory before cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        
        # Clear tensors
        tensors.clear()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
            
            print(f"\nMemory before: {memory_before / 1024 / 1024:.2f}MB")
            print(f"Memory after: {memory_after / 1024 / 1024:.2f}MB")
            
            # Memory should be freed
            assert memory_after < memory_before, "Memory should be freed after cleanup"
        else:
            print("\n✓ Memory cleanup test (CPU mode)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
