"""OpenTelemetry tracing configuration"""

from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

from app.core.config import settings
from app.core.logging import logger


def setup_tracing(app) -> Optional[trace.Tracer]:
    """
    Setup OpenTelemetry tracing with Jaeger exporter
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Tracer instance if tracing is enabled, None otherwise
    """
    if not settings.ENABLE_TRACING:
        logger.info("Tracing is disabled")
        return None
    
    try:
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: settings.PROJECT_NAME
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=settings.JAEGER_HOST,
            agent_port=settings.JAEGER_PORT,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument requests library
        RequestsInstrumentor().instrument()
        
        # Instrument Redis (if available)
        try:
            RedisInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Could not instrument Redis: {e}")
        
        logger.info(
            f"Tracing enabled - Jaeger endpoint: {settings.JAEGER_HOST}:{settings.JAEGER_PORT}"
        )
        
        # Get tracer
        tracer = trace.get_tracer(__name__)
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}", exc_info=True)
        return None


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance"""
    return trace.get_tracer(__name__)


def trace_function(name: str):
    """
    Decorator to trace a function
    
    Usage:
        @trace_function("my_function")
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def trace_async_function(name: str):
    """
    Decorator to trace an async function
    
    Usage:
        @trace_async_function("my_async_function")
        async def my_async_function():
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator
