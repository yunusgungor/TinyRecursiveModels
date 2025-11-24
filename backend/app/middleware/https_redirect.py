"""HTTPS enforcement middleware"""

from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.logging import logger


async def https_redirect_middleware(request: Request, call_next: Callable) -> Response:
    """Redirect HTTP requests to HTTPS in production"""
    
    # Skip in development mode
    if settings.DEBUG:
        return await call_next(request)
    
    # Check if request is HTTPS
    is_https = (
        request.url.scheme == "https" or
        request.headers.get("X-Forwarded-Proto") == "https" or
        request.headers.get("X-Forwarded-Ssl") == "on"
    )
    
    if not is_https:
        # Redirect to HTTPS
        https_url = request.url.replace(scheme="https")
        
        logger.info(
            f"Redirecting HTTP to HTTPS: {request.url} -> {https_url}",
            extra={
                "client_ip": request.client.host if request.client else "unknown",
                "path": request.url.path
            }
        )
        
        return RedirectResponse(
            url=str(https_url),
            status_code=status.HTTP_301_MOVED_PERMANENTLY
        )
    
    # Add security headers
    response = await call_next(request)
    
    # HSTS (HTTP Strict Transport Security)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # XSS protection
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'"
    )
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response
