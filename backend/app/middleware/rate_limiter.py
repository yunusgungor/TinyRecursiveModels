"""Rate limiting middleware"""

import time
from typing import Callable, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import ErrorResponse


class RateLimiter:
    """In-memory rate limiter (for production, use Redis)"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Unique identifier for the client (IP or user ID)
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        current_count = len(self.requests[client_id])
        
        if current_count >= self.max_requests:
            return False, 0
        
        # Add current request
        self.requests[client_id].append(now)
        
        remaining = self.max_requests - current_count - 1
        return True, remaining
    
    def get_reset_time(self, client_id: str) -> int:
        """Get time until rate limit resets (in seconds)"""
        if not self.requests[client_id]:
            return 0
        
        oldest_request = min(self.requests[client_id])
        reset_time = oldest_request + self.window_seconds
        remaining = max(0, int(reset_time - time.time()))
        
        return remaining


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60
)


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """Rate limiting middleware"""
    
    # Skip rate limiting if disabled (e.g., in tests)
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    # Get client identifier (IP address or user ID)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check if request is allowed
    is_allowed, remaining = rate_limiter.is_allowed(client_ip)
    
    if not is_allowed:
        reset_time = rate_limiter.get_reset_time(client_ip)
        
        logger.warning(
            f"Rate limit exceeded for client: {client_ip}",
            extra={
                "client_ip": client_ip,
                "path": request.url.path,
                "reset_in_seconds": reset_time
            }
        )
        
        error_response = ErrorResponse(
            error_code="RATE_LIMIT_EXCEEDED",
            message=f"Çok fazla istek gönderdiniz. Lütfen {reset_time} saniye sonra tekrar deneyin.",
            details={
                "retry_after_seconds": reset_time,
                "max_requests": rate_limiter.max_requests,
                "window_seconds": rate_limiter.window_seconds
            },
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, "request_id", "unknown")
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.model_dump(mode='json'),
            headers={
                "Retry-After": str(reset_time),
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset_time)
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + rate_limiter.window_seconds)
    
    return response
