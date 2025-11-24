"""
Performance tests for reasoning enhancement feature

These tests validate that reasoning generation has minimal performance impact
and meets the requirement of less than 10% overhead.

Requirements: 7.1
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock
import torch

from app.services.model_inference import ModelInferenceService
from app.services.reasoning_service import ReasoningService
from app.models.schemas import UserProfile, GiftItem


def create_test_user_profile(user_id: int = 1) -> UserProfile:
    """Create a test user profile"""
    return UserProfile(
        age=25 + (user_id % 50),
        hobbies=["cooking", "reading"],
        relationship="friend",
        budget=500.0,
        occasion="birthday",
        personality_traits=["practical", "creative"]
    )


def create_test_gift_items(count: int = 10) -> List[GiftItem]:
    """Create test gift items"""
    gifts = []
    categories = ["Kitchen", "Books", "Electronics", "Sports", "Fashion"]
    
    for i in range(count):
        gift = GiftItem(
            id=f"gift_{i}",
            name=f"Test Gift {i}",
            category=categories[i % len(categories)],
            price=50.0 + (i * 10),
            rating=4.0 + (i % 10) * 0.1,
            image_url=f"https://example.com/image_{i}.jpg",
            trendyol_url=f"https://trendyol.com/product/{i}",
            description=f"Test gift description {i}",
            tags=["test", "gift"],
            age_suitability=(18, 100),
            occasion_fit=["birthday", "anniversary"],
            in_stock=True
        )
        gifts.append(gift)
    
    return gifts


@pytest.fixture
def model_service(load_model):
    """Get model service with loaded model"""
    from app.services.model_inference import get_model_service
    service = get_model_service()
    
    if not service.is_loaded():
        pytest.skip("Model not loaded - skipping performance test")
    
    return service


@pytest.mark.asyncio
class TestReasoningPerformanceOverhead:
    """Test reasoning generation performance overhead"""
    
    async def test_inference_without_reasoning(self, model_service):
        """
        Measure baseline inference time without reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(20)
        
        # Warm up
        await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=5,
            include_reasoning=False
        )
        
        # Measure multiple runs
        times = []
        for _ in range(10):
            start_time = time.time()
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=False
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\n=== Inference WITHOUT Reasoning ===")
        print(f"Average time: {avg_time * 1000:.2f}ms")
        print(f"Std deviation: {std_dev * 1000:.2f}ms")
        print(f"Min time: {min(times) * 1000:.2f}ms")
        print(f"Max time: {max(times) * 1000:.2f}ms")
        
        # Store for comparison
        return avg_time
    
    async def test_inference_with_basic_reasoning(self, model_service):
        """
        Measure inference time with basic reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(20)
        
        # Warm up
        await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=5,
            include_reasoning=True,
            reasoning_level="basic"
        )
        
        # Measure multiple runs
        times = []
        for _ in range(10):
            start_time = time.time()
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="basic"
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\n=== Inference WITH Basic Reasoning ===")
        print(f"Average time: {avg_time * 1000:.2f}ms")
        print(f"Std deviation: {std_dev * 1000:.2f}ms")
        print(f"Min time: {min(times) * 1000:.2f}ms")
        print(f"Max time: {max(times) * 1000:.2f}ms")
        
        return avg_time
    
    async def test_inference_with_detailed_reasoning(self, model_service):
        """
        Measure inference time with detailed reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(20)
        
        # Warm up
        await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=5,
            include_reasoning=True,
            reasoning_level="detailed"
        )
        
        # Measure multiple runs
        times = []
        for _ in range(10):
            start_time = time.time()
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="detailed"
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\n=== Inference WITH Detailed Reasoning ===")
        print(f"Average time: {avg_time * 1000:.2f}ms")
        print(f"Std deviation: {std_dev * 1000:.2f}ms")
        print(f"Min time: {min(times) * 1000:.2f}ms")
        print(f"Max time: {max(times) * 1000:.2f}ms")
        
        return avg_time
    
    async def test_inference_with_full_reasoning(self, model_service):
        """
        Measure inference time with full reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(20)
        
        # Warm up
        await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=5,
            include_reasoning=True,
            reasoning_level="full"
        )
        
        # Measure multiple runs
        times = []
        for _ in range(10):
            start_time = time.time()
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="full"
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\n=== Inference WITH Full Reasoning ===")
        print(f"Average time: {avg_time * 1000:.2f}ms")
        print(f"Std deviation: {std_dev * 1000:.2f}ms")
        print(f"Min time: {min(times) * 1000:.2f}ms")
        print(f"Max time: {max(times) * 1000:.2f}ms")
        
        return avg_time
    
    async def test_reasoning_overhead_less_than_10_percent(self, model_service):
        """
        Verify that reasoning overhead is less than 10%
        
        This is the critical requirement from 7.1
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(20)
        
        # Measure baseline (no reasoning)
        baseline_times = []
        for _ in range(10):
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=False
            )
            baseline_times.append(time.time() - start_time)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # Measure with basic reasoning
        basic_times = []
        for _ in range(10):
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="basic"
            )
            basic_times.append(time.time() - start_time)
        
        basic_avg = statistics.mean(basic_times)
        
        # Measure with detailed reasoning
        detailed_times = []
        for _ in range(10):
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="detailed"
            )
            detailed_times.append(time.time() - start_time)
        
        detailed_avg = statistics.mean(detailed_times)
        
        # Measure with full reasoning
        full_times = []
        for _ in range(10):
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="full"
            )
            full_times.append(time.time() - start_time)
        
        full_avg = statistics.mean(full_times)
        
        # Calculate overhead percentages
        basic_overhead = ((basic_avg - baseline_avg) / baseline_avg) * 100
        detailed_overhead = ((detailed_avg - baseline_avg) / baseline_avg) * 100
        full_overhead = ((full_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"\n=== Reasoning Overhead Analysis ===")
        print(f"Baseline (no reasoning): {baseline_avg * 1000:.2f}ms")
        print(f"Basic reasoning: {basic_avg * 1000:.2f}ms (overhead: {basic_overhead:.2f}%)")
        print(f"Detailed reasoning: {detailed_avg * 1000:.2f}ms (overhead: {detailed_overhead:.2f}%)")
        print(f"Full reasoning: {full_avg * 1000:.2f}ms (overhead: {full_overhead:.2f}%)")
        
        # CRITICAL ASSERTION: Overhead must be less than 10%
        assert basic_overhead < 10.0, f"Basic reasoning overhead ({basic_overhead:.2f}%) exceeds 10% limit"
        assert detailed_overhead < 10.0, f"Detailed reasoning overhead ({detailed_overhead:.2f}%) exceeds 10% limit"
        assert full_overhead < 10.0, f"Full reasoning overhead ({full_overhead:.2f}%) exceeds 10% limit"
        
        print("\n✓ All reasoning levels meet the <10% overhead requirement")


@pytest.mark.asyncio
class TestReasoningConcurrentPerformance:
    """Test reasoning performance with concurrent requests"""
    
    async def test_concurrent_requests_without_reasoning(self, model_service):
        """
        Test concurrent requests without reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        num_concurrent = 10
        
        async def make_request(user_id: int):
            user_profile = create_test_user_profile(user_id)
            available_gifts = create_test_gift_items(20)
            
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=False
            )
            return time.time() - start_time
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_concurrent)]
        times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        avg_time = statistics.mean(times)
        throughput = num_concurrent / total_time
        
        print(f"\n=== {num_concurrent} Concurrent Requests WITHOUT Reasoning ===")
        print(f"Total time: {total_time * 1000:.2f}ms")
        print(f"Average request time: {avg_time * 1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} req/s")
        
        return avg_time, throughput
    
    async def test_concurrent_requests_with_reasoning(self, model_service):
        """
        Test concurrent requests with reasoning
        
        Validates: Requirements 7.1
        """
        service = model_service
        num_concurrent = 10
        
        async def make_request(user_id: int):
            user_profile = create_test_user_profile(user_id)
            available_gifts = create_test_gift_items(20)
            
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="detailed"
            )
            return time.time() - start_time
        
        # Run concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_concurrent)]
        times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        avg_time = statistics.mean(times)
        throughput = num_concurrent / total_time
        
        print(f"\n=== {num_concurrent} Concurrent Requests WITH Reasoning ===")
        print(f"Total time: {total_time * 1000:.2f}ms")
        print(f"Average request time: {avg_time * 1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} req/s")
        
        return avg_time, throughput
    
    async def test_concurrent_reasoning_overhead(self, model_service):
        """
        Verify reasoning overhead remains under 10% with concurrent requests
        
        Validates: Requirements 7.1
        """
        service = model_service
        num_concurrent = 10
        
        # Test without reasoning
        async def request_without_reasoning(user_id: int):
            user_profile = create_test_user_profile(user_id)
            available_gifts = create_test_gift_items(20)
            
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=False
            )
            return time.time() - start_time
        
        # Test with reasoning
        async def request_with_reasoning(user_id: int):
            user_profile = create_test_user_profile(user_id)
            available_gifts = create_test_gift_items(20)
            
            start_time = time.time()
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=5,
                include_reasoning=True,
                reasoning_level="detailed"
            )
            return time.time() - start_time
        
        # Run without reasoning
        tasks_without = [request_without_reasoning(i) for i in range(num_concurrent)]
        times_without = await asyncio.gather(*tasks_without)
        avg_without = statistics.mean(times_without)
        
        # Run with reasoning
        tasks_with = [request_with_reasoning(i) for i in range(num_concurrent)]
        times_with = await asyncio.gather(*tasks_with)
        avg_with = statistics.mean(times_with)
        
        # Calculate overhead
        overhead = ((avg_with - avg_without) / avg_without) * 100
        
        print(f"\n=== Concurrent Reasoning Overhead ===")
        print(f"Without reasoning: {avg_without * 1000:.2f}ms")
        print(f"With reasoning: {avg_with * 1000:.2f}ms")
        print(f"Overhead: {overhead:.2f}%")
        
        # Verify overhead is less than 10%
        assert overhead < 10.0, f"Concurrent reasoning overhead ({overhead:.2f}%) exceeds 10% limit"
        
        print("\n✓ Concurrent reasoning meets <10% overhead requirement")


@pytest.mark.asyncio
class TestLargeReasoningTraces:
    """Test performance with large reasoning traces"""
    
    async def test_large_reasoning_trace_generation(self, model_service):
        """
        Test reasoning generation with large traces
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        
        # Create many gift items to generate large reasoning traces
        available_gifts = create_test_gift_items(100)
        
        start_time = time.time()
        recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=10,
            include_reasoning=True,
            reasoning_level="full"
        )
        elapsed = time.time() - start_time
        
        print(f"\n=== Large Reasoning Trace (100 gifts) ===")
        print(f"Generation time: {elapsed * 1000:.2f}ms")
        
        if reasoning_trace:
            # Estimate trace size
            import json
            trace_json = json.dumps(reasoning_trace, default=str)
            trace_size_kb = len(trace_json) / 1024
            
            print(f"Trace size: {trace_size_kb:.2f}KB")
            
            # Verify trace is not excessively large
            assert trace_size_kb < 500, "Reasoning trace should be under 500KB"
        
        # Large traces should still complete in reasonable time
        assert elapsed < 5.0, "Large reasoning trace generation should complete in under 5 seconds"
    
    async def test_reasoning_trace_serialization_performance(self, model_service):
        """
        Test JSON serialization performance of reasoning traces
        
        Validates: Requirements 7.1
        """
        service = model_service
        user_profile = create_test_user_profile()
        available_gifts = create_test_gift_items(50)
        
        # Generate reasoning trace
        recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
            user_profile=user_profile,
            available_gifts=available_gifts,
            max_recommendations=10,
            include_reasoning=True,
            reasoning_level="full"
        )
        
        if reasoning_trace:
            import json
            
            # Measure serialization time
            start_time = time.time()
            trace_json = json.dumps(reasoning_trace, default=str)
            serialization_time = time.time() - start_time
            
            print(f"\n=== Reasoning Trace Serialization ===")
            print(f"Serialization time: {serialization_time * 1000:.2f}ms")
            print(f"Trace size: {len(trace_json) / 1024:.2f}KB")
            
            # Serialization should be fast
            assert serialization_time < 0.1, "Serialization should be under 100ms"
    
    async def test_reasoning_memory_usage(self, model_service):
        """
        Test memory usage with reasoning traces
        
        Validates: Requirements 7.1
        """
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        service = model_service
        
        # Generate multiple reasoning traces
        for i in range(10):
            user_profile = create_test_user_profile(i)
            available_gifts = create_test_gift_items(50)
            
            await service.generate_recommendations(
                user_profile=user_profile,
                available_gifts=available_gifts,
                max_recommendations=10,
                include_reasoning=True,
                reasoning_level="full"
            )
        
        # Get final memory
        gc.collect()
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - baseline_memory
        
        print(f"\n=== Reasoning Memory Usage ===")
        print(f"Baseline memory: {baseline_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 200, "Memory increase should be under 200MB for 10 reasoning traces"


@pytest.mark.asyncio
class TestReasoningComponentPerformance:
    """Test individual reasoning component performance"""
    
    async def test_tool_selection_reasoning_performance(self):
        """
        Test tool selection reasoning generation performance
        
        Validates: Requirements 7.1
        """
        reasoning_service = ReasoningService()
        user_profile = create_test_user_profile()
        
        # Mock tool selection trace
        tool_selection_trace = {
            "price_comparison": {
                "selected": True,
                "score": 0.85,
                "reason": "Budget constraint",
                "confidence": 0.85,
                "priority": 1,
                "factors": {"budget_constraint": 0.9}
            },
            "review_analysis": {
                "selected": True,
                "score": 0.75,
                "reason": "Quality preference",
                "confidence": 0.75,
                "priority": 2,
                "factors": {"quality_preference": 0.8}
            }
        }
        
        # Measure generation time
        times = []
        for _ in range(100):
            start_time = time.time()
            reasoning = reasoning_service.generate_tool_selection_reasoning(
                tool_selection_trace,
                user_profile
            )
            times.append(time.time() - start_time)
        
        avg_time = statistics.mean(times)
        
        print(f"\n=== Tool Selection Reasoning Performance ===")
        print(f"Average time: {avg_time * 1000:.4f}ms")
        print(f"Max time: {max(times) * 1000:.4f}ms")
        
        # Should be very fast
        assert avg_time < 0.001, "Tool selection reasoning should be under 1ms"
    
    async def test_category_reasoning_performance(self):
        """
        Test category reasoning generation performance
        
        Validates: Requirements 7.1
        """
        reasoning_service = ReasoningService()
        user_profile = create_test_user_profile()
        
        # Mock category trace
        category_trace = {
            "Kitchen": {
                "score": 0.85,
                "reasons": ["Hobby match: cooking"],
                "feature_contributions": {"hobby_match": 0.45}
            },
            "Books": {
                "score": 0.75,
                "reasons": ["Hobby match: reading"],
                "feature_contributions": {"hobby_match": 0.40}
            }
        }
        
        # Measure generation time
        times = []
        for _ in range(100):
            start_time = time.time()
            reasoning = reasoning_service.generate_category_reasoning(
                category_trace,
                user_profile
            )
            times.append(time.time() - start_time)
        
        avg_time = statistics.mean(times)
        
        print(f"\n=== Category Reasoning Performance ===")
        print(f"Average time: {avg_time * 1000:.4f}ms")
        print(f"Max time: {max(times) * 1000:.4f}ms")
        
        # Should be very fast
        assert avg_time < 0.001, "Category reasoning should be under 1ms"
    
    async def test_gift_reasoning_performance(self):
        """
        Test gift reasoning generation performance
        
        Validates: Requirements 7.1
        """
        reasoning_service = ReasoningService()
        user_profile = create_test_user_profile()
        gift = create_test_gift_items(1)[0]
        
        model_output = {
            "category_scores": {"Kitchen": 0.85}
        }
        
        tool_results = {
            "review_analysis": {"average_rating": 4.5},
            "trend_analysis": {"trending": []}
        }
        
        # Measure generation time
        times = []
        for _ in range(100):
            start_time = time.time()
            reasoning = reasoning_service.generate_gift_reasoning(
                gift,
                user_profile,
                model_output,
                tool_results
            )
            times.append(time.time() - start_time)
        
        avg_time = statistics.mean(times)
        
        print(f"\n=== Gift Reasoning Performance ===")
        print(f"Average time: {avg_time * 1000:.4f}ms")
        print(f"Max time: {max(times) * 1000:.4f}ms")
        
        # Should be very fast
        assert avg_time < 0.002, "Gift reasoning should be under 2ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "asyncio"])
