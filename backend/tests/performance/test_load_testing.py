"""
Load testing for Trendyol Gift Recommendation API

This module contains performance tests to validate system behavior under load.
Tests concurrent users, response times, and resource utilization.

Requirements: 6.4, 6.5
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import requests
from datetime import datetime

from app.models.schemas import UserProfile


# Test configuration
BASE_URL = "http://localhost:8000"
API_V1_PREFIX = "/api/v1"


class LoadTestMetrics:
    """Container for load test metrics"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.start_time: float = 0
        self.end_time: float = 0
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
    
    def add_response_time(self, response_time: float):
        """Add a response time measurement"""
        self.response_times.append(response_time)
    
    def increment_success(self):
        """Increment success counter"""
        self.success_count += 1
    
    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1
    
    def add_cpu_sample(self, cpu_percent: float):
        """Add CPU utilization sample"""
        self.cpu_samples.append(cpu_percent)
    
    def add_memory_sample(self, memory_mb: float):
        """Add memory utilization sample"""
        self.memory_samples.append(memory_mb)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.response_times:
            return {
                "error": "No response times recorded",
                "success_count": self.success_count,
                "error_count": self.error_count,
            }
        
        total_time = self.end_time - self.start_time
        throughput = self.success_count / total_time if total_time > 0 else 0
        
        return {
            "total_requests": self.success_count + self.error_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / (self.success_count + self.error_count) * 100,
            "total_duration_seconds": total_time,
            "throughput_requests_per_second": throughput,
            "response_times": {
                "min_ms": min(self.response_times) * 1000,
                "max_ms": max(self.response_times) * 1000,
                "mean_ms": statistics.mean(self.response_times) * 1000,
                "median_ms": statistics.median(self.response_times) * 1000,
                "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000 if len(self.response_times) >= 20 else max(self.response_times) * 1000,
                "p99_ms": statistics.quantiles(self.response_times, n=100)[98] * 1000 if len(self.response_times) >= 100 else max(self.response_times) * 1000,
            },
            "resource_utilization": {
                "cpu_percent": {
                    "min": min(self.cpu_samples) if self.cpu_samples else 0,
                    "max": max(self.cpu_samples) if self.cpu_samples else 0,
                    "mean": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
                },
                "memory_mb": {
                    "min": min(self.memory_samples) if self.memory_samples else 0,
                    "max": max(self.memory_samples) if self.memory_samples else 0,
                    "mean": statistics.mean(self.memory_samples) if self.memory_samples else 0,
                },
            }
        }


def create_test_profile(user_id: int) -> Dict[str, Any]:
    """
    Create a test user profile
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        User profile dictionary
    """
    hobbies = ["reading", "gaming", "cooking", "sports", "music"]
    relationships = ["friend", "mother", "father", "sibling", "partner"]
    occasions = ["birthday", "anniversary", "graduation", "holiday"]
    
    return {
        "age": 25 + (user_id % 50),
        "hobbies": [hobbies[user_id % len(hobbies)], hobbies[(user_id + 1) % len(hobbies)]],
        "relationship": relationships[user_id % len(relationships)],
        "budget": 100 + (user_id % 900),
        "occasion": occasions[user_id % len(occasions)],
        "personality_traits": ["practical", "creative"],
    }


def make_recommendation_request(user_id: int, session: requests.Session) -> float:
    """
    Make a single recommendation request
    
    Args:
        user_id: User identifier
        session: Requests session
        
    Returns:
        Response time in seconds
        
    Raises:
        Exception: If request fails
    """
    profile = create_test_profile(user_id)
    
    payload = {
        "user_profile": profile,
        "max_recommendations": 5,
        "use_cache": True,
    }
    
    start_time = time.time()
    response = session.post(
        f"{BASE_URL}{API_V1_PREFIX}/recommendations",
        json=payload,
        timeout=10
    )
    response_time = time.time() - start_time
    
    response.raise_for_status()
    return response_time


@pytest.mark.performance
class TestConcurrentUsers:
    """Test system behavior with concurrent users"""
    
    def test_10_concurrent_users(self):
        """
        Test with 10 concurrent users
        
        Validates: Requirements 6.4
        """
        num_users = 10
        requests_per_user = 5
        
        metrics = self._run_concurrent_test(num_users, requests_per_user)
        summary = metrics.get_summary()
        
        print("\n=== 10 Concurrent Users Test ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Mean Response Time: {summary['response_times']['mean_ms']:.2f}ms")
        print(f"P95 Response Time: {summary['response_times']['p95_ms']:.2f}ms")
        print(f"Throughput: {summary['throughput_requests_per_second']:.2f} req/s")
        
        # Assertions
        assert summary['success_rate'] >= 95, "Success rate should be at least 95%"
        assert summary['response_times']['mean_ms'] < 3000, "Mean response time should be under 3 seconds"
        assert summary['response_times']['p95_ms'] < 5000, "P95 response time should be under 5 seconds"
    
    def test_50_concurrent_users(self):
        """
        Test with 50 concurrent users
        
        Validates: Requirements 6.4
        """
        num_users = 50
        requests_per_user = 3
        
        metrics = self._run_concurrent_test(num_users, requests_per_user)
        summary = metrics.get_summary()
        
        print("\n=== 50 Concurrent Users Test ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Mean Response Time: {summary['response_times']['mean_ms']:.2f}ms")
        print(f"P95 Response Time: {summary['response_times']['p95_ms']:.2f}ms")
        print(f"Throughput: {summary['throughput_requests_per_second']:.2f} req/s")
        
        # Assertions - more lenient for higher load
        assert summary['success_rate'] >= 90, "Success rate should be at least 90%"
        assert summary['response_times']['mean_ms'] < 5000, "Mean response time should be under 5 seconds"
    
    def _run_concurrent_test(self, num_users: int, requests_per_user: int) -> LoadTestMetrics:
        """
        Run concurrent user test
        
        Args:
            num_users: Number of concurrent users
            requests_per_user: Requests per user
            
        Returns:
            LoadTestMetrics with results
        """
        metrics = LoadTestMetrics()
        metrics.start_time = time.time()
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(
            self._monitor_resources(metrics, duration=num_users * requests_per_user * 0.5)
        )
        
        # Create thread pool for concurrent requests
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            # Create sessions for each user
            sessions = [requests.Session() for _ in range(num_users)]
            
            # Submit all requests
            futures = []
            for user_id in range(num_users):
                for _ in range(requests_per_user):
                    future = executor.submit(
                        make_recommendation_request,
                        user_id,
                        sessions[user_id]
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    response_time = future.result()
                    metrics.add_response_time(response_time)
                    metrics.increment_success()
                except Exception as e:
                    print(f"Request failed: {str(e)}")
                    metrics.increment_error()
            
            # Close sessions
            for session in sessions:
                session.close()
        
        metrics.end_time = time.time()
        
        return metrics
    
    async def _monitor_resources(self, metrics: LoadTestMetrics, duration: float):
        """
        Monitor system resources during test
        
        Args:
            metrics: Metrics container
            duration: Monitoring duration in seconds
        """
        start = time.time()
        process = psutil.Process()
        
        while time.time() - start < duration:
            try:
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                metrics.add_cpu_sample(cpu_percent)
                
                # Get memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                metrics.add_memory_sample(memory_mb)
                
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Resource monitoring error: {str(e)}")
                break


@pytest.mark.performance
class TestResponseTimes:
    """Test API response time requirements"""
    
    def test_cached_response_time(self):
        """
        Test that cached responses are fast
        
        Validates: Requirements 6.1, 6.4
        """
        profile = create_test_profile(1)
        payload = {
            "user_profile": profile,
            "max_recommendations": 5,
            "use_cache": True,
        }
        
        session = requests.Session()
        
        # First request (cache miss)
        response1 = session.post(
            f"{BASE_URL}{API_V1_PREFIX}/recommendations",
            json=payload,
            timeout=10
        )
        assert response1.status_code == 200
        
        # Second request (cache hit)
        start_time = time.time()
        response2 = session.post(
            f"{BASE_URL}{API_V1_PREFIX}/recommendations",
            json=payload,
            timeout=10
        )
        cached_response_time = time.time() - start_time
        
        assert response2.status_code == 200
        data = response2.json()
        
        print(f"\nCached Response Time: {cached_response_time * 1000:.2f}ms")
        print(f"Cache Hit: {data.get('cache_hit', False)}")
        
        # Cached responses should be very fast
        assert cached_response_time < 0.5, "Cached response should be under 500ms"
        assert data.get('cache_hit') == True, "Second request should be cache hit"
        
        session.close()
    
    def test_uncached_response_time(self):
        """
        Test that uncached responses meet SLA
        
        Validates: Requirements 6.4
        """
        # Use unique profile to avoid cache
        profile = create_test_profile(int(time.time()))
        payload = {
            "user_profile": profile,
            "max_recommendations": 5,
            "use_cache": False,  # Disable cache
        }
        
        session = requests.Session()
        
        start_time = time.time()
        response = session.post(
            f"{BASE_URL}{API_V1_PREFIX}/recommendations",
            json=payload,
            timeout=10
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        
        print(f"\nUncached Response Time: {response_time * 1000:.2f}ms")
        
        # Uncached responses should complete within 3 seconds
        assert response_time < 3.0, "Uncached response should be under 3 seconds"
        
        session.close()


@pytest.mark.performance
class TestResourceUtilization:
    """Test system resource utilization under load"""
    
    def test_memory_usage_under_load(self):
        """
        Test that memory usage stays within limits
        
        Validates: Requirements 6.5
        """
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        # Run load test
        num_requests = 20
        session = requests.Session()
        
        for i in range(num_requests):
            profile = create_test_profile(i)
            payload = {
                "user_profile": profile,
                "max_recommendations": 5,
                "use_cache": True,
            }
            
            try:
                response = session.post(
                    f"{BASE_URL}{API_V1_PREFIX}/recommendations",
                    json=payload,
                    timeout=10
                )
                assert response.status_code == 200
            except Exception as e:
                print(f"Request {i} failed: {str(e)}")
        
        session.close()
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - baseline_memory
        
        print(f"\nBaseline Memory: {baseline_memory:.2f}MB")
        print(f"Final Memory: {final_memory:.2f}MB")
        print(f"Memory Increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500, "Memory increase should be under 500MB"
    
    def test_cpu_usage_under_load(self):
        """
        Test CPU usage during inference
        
        Validates: Requirements 6.5
        """
        process = psutil.Process()
        cpu_samples = []
        
        session = requests.Session()
        
        # Make requests and sample CPU
        for i in range(10):
            profile = create_test_profile(i)
            payload = {
                "user_profile": profile,
                "max_recommendations": 5,
                "use_cache": False,  # Force inference
            }
            
            try:
                response = session.post(
                    f"{BASE_URL}{API_V1_PREFIX}/recommendations",
                    json=payload,
                    timeout=10
                )
                
                # Sample CPU after request
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                
            except Exception as e:
                print(f"Request {i} failed: {str(e)}")
        
        session.close()
        
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            print(f"\nAverage CPU: {avg_cpu:.2f}%")
            print(f"Max CPU: {max_cpu:.2f}%")
            
            # CPU usage should be reasonable
            # Note: This is process-level, not system-level
            assert avg_cpu < 200, "Average CPU usage should be reasonable"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
