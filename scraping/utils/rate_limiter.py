"""
Rate Limiter for Web Scraping
Controls request rate and implements anti-bot strategies
"""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List


class RateLimiter:
    """Controls request rate and implements anti-bot strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rate limiter
        
        Args:
            config: Rate limit configuration dictionary
        """
        self.requests_per_minute = config.get('requests_per_minute', 20)
        self.delay_range = config.get('delay_between_requests', [2, 5])
        self.max_concurrent = config.get('max_concurrent_requests', 10)
        
        # Track request times
        self.request_times: List[datetime] = []
        
        # Semaphore for concurrent request control
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"RateLimiter initialized: {self.requests_per_minute} req/min, "
            f"{self.max_concurrent} concurrent, delay {self.delay_range}s"
        )
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request
        Blocks until rate limit allows the request
        """
        await self.semaphore.acquire()
        await self._wait_if_needed()

    def release(self) -> None:
        """Release the semaphore after request completes"""
        self.semaphore.release()
    
    async def _wait_if_needed(self) -> None:
        """
        Wait if rate limit is exceeded
        Implements both rate limiting and random delays
        """
        # Clean old requests (older than 1 minute)
        cutoff = datetime.now() - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Check if we need to wait for rate limit
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (datetime.now() - oldest_request).total_seconds()
            
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                # Clean again after waiting
                cutoff = datetime.now() - timedelta(minutes=1)
                self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Random delay to appear more human-like
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        self.logger.debug(f"Random delay: {delay:.2f}s")
        await asyncio.sleep(delay)
        
        # Record this request
        self.request_times.append(datetime.now())
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with current stats
        """
        cutoff = datetime.now() - timedelta(minutes=1)
        recent_requests = [t for t in self.request_times if t > cutoff]
        
        return {
            'requests_last_minute': len(recent_requests),
            'requests_per_minute_limit': self.requests_per_minute,
            'available_slots': self.semaphore._value,
            'max_concurrent': self.max_concurrent
        }
