"""Prometheus metrics endpoint"""

from fastapi import APIRouter, Response
from app.services.monitoring_service import monitoring_service

router = APIRouter()


@router.get("/metrics", response_class=Response)
async def get_metrics() -> Response:
    """
    Export metrics in Prometheus format
    
    Returns metrics for:
    - Tool usage and success rates
    - Model inference times
    - API response times
    - System resources (CPU, memory, GPU)
    """
    metrics_text = monitoring_service.export_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain")
