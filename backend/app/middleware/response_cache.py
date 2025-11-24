"""Response caching middleware for API endpoints"""

import hashlib
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware to cache API responses for GET requests
    """
    
    def __init__(self, app, cache_service=None, ttl: int = 300):
        """
        Initialize response cache middleware
        
        Args:
            app: FastAPI application
            cache_service: Cache service instance
            ttl: Time to live for cached responses in seconds
        """
        super().__init__(app)
        self.cache_service = cache_service
        self.ttl = ttl
        self.cacheable_paths = [
            "/api/v1/health",
            "/api/v1/tools/stats",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and cache response if applicable
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response (cached or fresh)
        """
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if path is cacheable
        path = request.url.path
        if not any(path.startswith(cacheable) for cacheable in self.cacheable_paths):
            return await call_next(request)
        
        # Skip if no cache service
        if not self.cache_service:
            return await call_next(request)
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Try to get from cache
            cached_response = await self.cache_service.get_trendyol_data(
                f"response:{cache_key}"
            )
            
            if cached_response:
                logger.debug(f"Cache hit for {path}")
                return JSONResponse(
                    content=cached_response,
                    headers={"X-Cache": "HIT"}
                )
            
            # Get fresh response
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Parse JSON
                try:
                    response_data = json.loads(body.decode())
                    
                    # Cache the response
                    await self.cache_service.set_trendyol_data(
                        f"response:{cache_key}",
                        response_data,
                        ttl=self.ttl
                    )
                    
                    logger.debug(f"Cached response for {path}")
                    
                    # Return response with cache miss header
                    return JSONResponse(
                        content=response_data,
                        headers={"X-Cache": "MISS"}
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response for caching: {path}")
                    return response
            
            return response
            
        except Exception as e:
            logger.error(f"Error in response cache middleware: {str(e)}")
            return await call_next(request)
    
    def _generate_cache_key(self, request: Request) -> str:
        """
        Generate cache key from request
        
        Args:
            request: Incoming request
            
        Returns:
            Cache key string
        """
        # Include path and query params in cache key
        key_parts = [
            request.url.path,
            str(sorted(request.query_params.items()))
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
