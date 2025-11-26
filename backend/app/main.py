"""FastAPI application entry point"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import health, recommendations, tools, metrics
from app.core.config import settings
from app.core.exceptions import BaseAPIException
from app.core.logging import logger
from app.middleware.error_handler import error_handler_middleware
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.session import session_middleware
from app.middleware.https_redirect import https_redirect_middleware
from app.models.schemas import ErrorResponse
from datetime import datetime
import uuid


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events"""
    # Startup
    logger.info("Starting Trendyol Gift Recommendation API")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    
    # Setup tracing
    try:
        from app.core.tracing import setup_tracing
        setup_tracing(app)
    except ImportError:
        logger.warning("OpenTelemetry not installed, tracing disabled")
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
    
    # Start resource monitoring background task
    import asyncio
    from app.services.monitoring_service import monitoring_service
    
    async def monitor_resources():
        """Background task to monitor resource usage"""
        while True:
            try:
                # Check resource limits every 30 seconds
                await asyncio.sleep(30)
                monitoring_service.check_resource_limits()
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    monitoring_task = asyncio.create_task(monitor_resources())
    
    yield
    
    # Shutdown
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    logger.info("Shutting down Trendyol Gift Recommendation API")


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (security)
    if not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly in production
        )
    
    # Security middlewares
    app.middleware("http")(https_redirect_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(session_middleware)
    
    # Error handling middleware
    app.middleware("http")(error_handler_middleware)
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        errors = exc.errors()
        field = errors[0]["loc"][-1] if errors else "unknown"
        message = f"Geçersiz değer: {field}"
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        logger.error(
            "Validation Error",
            extra={
                "errors": errors,
                "request_id": request_id,
                "path": request.url.path,
            }
        )
        
        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=message,
            details={"field": field, "validation_errors": errors},
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(BaseAPIException)
    async def api_exception_handler(request: Request, exc: BaseAPIException):
        """Handle custom API exceptions"""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        logger.error(
            f"API Error: {exc.error_code}",
            extra={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True
        )
        
        error_response = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump(mode='json')
        )
    
    # Include routers
    app.include_router(
        health.router,
        prefix=settings.API_V1_PREFIX,
        tags=["health"]
    )
    app.include_router(
        recommendations.router,
        prefix=settings.API_V1_PREFIX,
        tags=["recommendations"]
    )
    app.include_router(
        tools.router,
        prefix=settings.API_V1_PREFIX,
        tags=["tools"]
    )
    # Metrics endpoints are in health router
    # app.include_router(
    #     metrics.router,
    #     prefix=settings.API_V1_PREFIX,
    #     tags=["metrics"]
    # )
    
    @app.get("/")
    async def root() -> dict:
        """Root endpoint"""
        return {
            "message": "Trendyol Gift Recommendation API",
            "version": settings.VERSION,
            "docs": f"{settings.API_V1_PREFIX}/docs"
        }
    
    return app


app = create_application()
