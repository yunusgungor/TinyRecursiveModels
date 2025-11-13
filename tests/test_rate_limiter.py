"""
Unit tests for RateLimiter
"""

import pytest
import asyncio
from datetime import datetime

from scraping.utils.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_delay():
    """Test that rate limiter adds delay"""
    config = {
        'requests_per_minute': 60,
        'delay_between_requests': [0.1, 0.2],
        'max_concurrent_requests': 5
    }
    
    rate_limiter = RateLimiter(config)
    
    start_time = datetime.now()
    await rate_limiter.acquire()
    rate_limiter.release()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    assert duration >= 0.1  # Should have at least minimum delay


@pytest.mark.asyncio
async def test_rate_limiter_concurrent():
    """Test concurrent request limiting"""
    config = {
        'requests_per_minute': 60,
        'delay_between_requests': [0.01, 0.02],
        'max_concurrent_requests': 2
    }
    
    rate_limiter = RateLimiter(config)
    
    async def make_request():
        async with rate_limiter:
            await asyncio.sleep(0.1)
    
    # Try to make 3 concurrent requests (limit is 2)
    start_time = datetime.now()
    await asyncio.gather(make_request(), make_request(), make_request())
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    # Should take longer than 0.1s due to concurrent limit
    assert duration > 0.1


def test_rate_limiter_stats():
    """Test rate limiter statistics"""
    config = {
        'requests_per_minute': 20,
        'delay_between_requests': [1, 2],
        'max_concurrent_requests': 10
    }
    
    rate_limiter = RateLimiter(config)
    stats = rate_limiter.get_stats()
    
    assert 'requests_last_minute' in stats
    assert 'requests_per_minute_limit' in stats
    assert stats['requests_per_minute_limit'] == 20
