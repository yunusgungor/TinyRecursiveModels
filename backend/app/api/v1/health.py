"""Health check endpoints"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Response

from app.models.schemas import HealthResponse
from app.services.monitoring_service import monitoring_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns system health status including:
    - Overall status
    - Model loading status
    - Trendyol API status
    - Cache status
    """
    health_status = monitoring_service.get_health_status()
    
    return HealthResponse(
        status=health_status["status"],
        model_loaded=health_status["model_loaded"],
        trendyol_api_status=health_status["trendyol_api_status"],
        cache_status=health_status["cache_status"],
        timestamp=health_status["timestamp"]
    )


@router.get("/metrics")
async def get_metrics() -> Response:
    """
    Get Prometheus-formatted metrics
    
    Returns metrics in Prometheus exposition format for scraping
    """
    metrics_text = monitoring_service.export_prometheus_metrics()
    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4"
    )


@router.get("/resources")
async def get_resources() -> Dict[str, Any]:
    """
    Get system resource usage
    
    Returns:
    - CPU usage
    - Memory usage
    - GPU usage (if available)
    """
    return monitoring_service.get_resource_metrics()


@router.get("/performance")
async def get_performance() -> Dict[str, Any]:
    """
    Get performance metrics
    
    Returns:
    - Average inference time
    - Average response time
    - Total inferences
    - Total requests
    """
    return monitoring_service.get_performance_metrics()
