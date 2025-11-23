"""Tool statistics endpoints"""

from typing import Dict, Any
from fastapi import APIRouter

from app.models.schemas import ToolStatsResponse
from app.services.monitoring_service import monitoring_service

router = APIRouter()


@router.get("/tools/stats", response_model=ToolStatsResponse)
async def get_tool_stats() -> ToolStatsResponse:
    """
    Get tool usage statistics
    
    Returns:
    - Tool usage counts
    - Success rates
    - Average execution times
    """
    stats = monitoring_service.metrics_collector.get_tool_stats()
    
    return ToolStatsResponse(
        tool_usage=stats["tool_usage"],
        success_rates=stats["success_rates"],
        average_execution_times=stats["average_execution_times"]
    )


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics
    
    Returns:
    - Average inference time
    - Average response time
    - Total inferences
    - Total requests
    """
    return monitoring_service.metrics_collector.get_performance_metrics()
