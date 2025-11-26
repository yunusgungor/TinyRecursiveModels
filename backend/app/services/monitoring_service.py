"""Monitoring and metrics collection service"""

import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.core.logging import logger


class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self):
        self._tool_usage: Dict[str, int] = defaultdict(int)
        self._tool_success: Dict[str, int] = defaultdict(int)
        self._tool_failures: Dict[str, int] = defaultdict(int)
        self._tool_execution_times: Dict[str, list] = defaultdict(list)
        self._inference_times: list = []
        self._api_response_times: list = []
        self._lock = threading.Lock()
        
    def record_tool_execution(
        self,
        tool_name: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Record tool execution metrics"""
        with self._lock:
            self._tool_usage[tool_name] += 1
            if success:
                self._tool_success[tool_name] += 1
            else:
                self._tool_failures[tool_name] += 1
            self._tool_execution_times[tool_name].append(execution_time)
    
    def record_inference_time(self, inference_time: float) -> None:
        """Record model inference time"""
        with self._lock:
            self._inference_times.append(inference_time)
            # Keep only last 1000 measurements
            if len(self._inference_times) > 1000:
                self._inference_times = self._inference_times[-1000:]
    
    def record_api_response_time(self, response_time: float) -> None:
        """Record API response time"""
        with self._lock:
            self._api_response_times.append(response_time)
            # Keep only last 1000 measurements
            if len(self._api_response_times) > 1000:
                self._api_response_times = self._api_response_times[-1000:]
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        with self._lock:
            success_rates = {}
            avg_execution_times = {}
            
            for tool_name in self._tool_usage:
                total = self._tool_usage[tool_name]
                successes = self._tool_success[tool_name]
                success_rates[tool_name] = successes / total if total > 0 else 0.0
                
                times = self._tool_execution_times[tool_name]
                avg_execution_times[tool_name] = (
                    sum(times) / len(times) if times else 0.0
                )
            
            return {
                "tool_usage": dict(self._tool_usage),
                "success_rates": success_rates,
                "average_execution_times": avg_execution_times
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            avg_inference_time = (
                sum(self._inference_times) / len(self._inference_times)
                if self._inference_times else 0.0
            )
            avg_response_time = (
                sum(self._api_response_times) / len(self._api_response_times)
                if self._api_response_times else 0.0
            )
            
            return {
                "average_inference_time": avg_inference_time,
                "average_response_time": avg_response_time,
                "total_inferences": len(self._inference_times),
                "total_requests": len(self._api_response_times)
            }


class MonitoringService:
    """Service for monitoring system health and resources"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self._model_loaded = False
        self._model_load_time: Optional[datetime] = None
        self._container_start_time = datetime.utcnow()
        self._restart_count = 0
        self._restart_history: list = []  # Store last 10 restarts
        self._health_check_failures: Dict[str, int] = defaultdict(int)
        self._last_health_check_errors: Dict[str, Dict[str, Any]] = {}
    
    def set_model_loaded(self, loaded: bool) -> None:
        """Set model loaded status"""
        self._model_loaded = loaded
        if loaded:
            self._model_load_time = datetime.utcnow()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            "loaded": self._model_loaded,
            "load_time": self._model_load_time.isoformat() if self._model_load_time else None
        }
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "percent": memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                "total_mb": 0,
                "used_mb": 0,
                "available_mb": 0,
                "percent": 0
            }
    
    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage information"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if not torch.cuda.is_available():
                return None
            
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                
                gpu_info.append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": memory_allocated,
                    "memory_reserved_mb": memory_reserved,
                })
            
            return {"devices": gpu_info} if gpu_info else None
        except Exception as e:
            logger.error(f"Error getting GPU usage: {e}")
            return None
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get all resource metrics"""
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "gpu": self.get_gpu_usage(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def record_health_check_failure(
        self,
        check_name: str,
        endpoint: str,
        status_code: Optional[int],
        error_message: str,
        response_body: Optional[str] = None
    ) -> None:
        """Record a health check failure with detailed information"""
        self._health_check_failures[check_name] += 1
        
        error_details = {
            "check_name": check_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "error_message": error_message,
            "response_body": response_body[:500] if response_body else None,  # Limit size
            "timestamp": datetime.utcnow().isoformat(),
            "failure_count": self._health_check_failures[check_name]
        }
        
        self._last_health_check_errors[check_name] = error_details
        
        logger.error(
            f"Health check failed: {check_name}",
            extra=error_details
        )
    
    def check_cache_status(self) -> str:
        """Check cache service status"""
        check_name = "cache"
        endpoint = "redis://redis:6379"
        
        try:
            # TODO: Implement actual cache health check
            # For now, return healthy as placeholder
            return "healthy"
        except Exception as e:
            self.record_health_check_failure(
                check_name=check_name,
                endpoint=endpoint,
                status_code=None,
                error_message=str(e)
            )
            return "unhealthy"
    
    def check_trendyol_api_status(self) -> str:
        """Check Trendyol API status"""
        check_name = "trendyol_api"
        endpoint = "https://api.trendyol.com"
        
        try:
            # TODO: Implement actual API health check
            # For now, return healthy as placeholder
            return "healthy"
        except Exception as e:
            self.record_health_check_failure(
                check_name=check_name,
                endpoint=endpoint,
                status_code=None,
                error_message=str(e)
            )
            return "unhealthy"
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get system resource usage metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_info = {
            "total_mb": round(memory.total / (1024 * 1024), 2),
            "used_mb": round(memory.used / (1024 * 1024), 2),
            "available_mb": round(memory.available / (1024 * 1024), 2),
            "percent": memory.percent
        }
        
        # GPU usage (if available)
        gpu_info = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_devices = []
                for i in range(torch.cuda.device_count()):
                    gpu_devices.append({
                        "device_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_mb": round(
                            torch.cuda.memory_allocated(i) / (1024 * 1024), 2
                        ),
                        "memory_reserved_mb": round(
                            torch.cuda.memory_reserved(i) / (1024 * 1024), 2
                        )
                    })
                gpu_info = {"devices": gpu_devices}
            except Exception as e:
                logger.error(f"Failed to get GPU metrics: {e}")
        
        return {
            "cpu_percent": cpu_percent,
            "memory": memory_info,
            "gpu": gpu_info,
            "timestamp": datetime.utcnow()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics_collector.get_performance_metrics()
    
    def record_container_restart(self, reason: str, previous_state: Optional[str] = None) -> None:
        """Record a container restart event with detailed information"""
        self._restart_count += 1
        previous_uptime = (datetime.utcnow() - self._container_start_time).total_seconds()
        
        restart_event = {
            "restart_count": self._restart_count,
            "reason": reason,
            "previous_state": previous_state or "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "previous_uptime_seconds": previous_uptime
        }
        
        # Store in history (keep last 10)
        self._restart_history.append(restart_event)
        if len(self._restart_history) > 10:
            self._restart_history = self._restart_history[-10:]
        
        logger.warning(
            "Container restart detected",
            extra=restart_event
        )
        
        self._container_start_time = datetime.utcnow()
    
    def get_restart_history(self) -> Dict[str, Any]:
        """Get container restart history"""
        return {
            "total_restarts": self._restart_count,
            "restart_events": self._restart_history,
            "current_uptime_seconds": (
                datetime.utcnow() - self._container_start_time
            ).total_seconds()
        }
    
    def check_resource_limits(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0
    ) -> Dict[str, Any]:
        """
        Check if resource usage is approaching limits and emit warnings
        
        Args:
            cpu_threshold: CPU usage percentage threshold (default: 80%)
            memory_threshold: Memory usage percentage threshold (default: 80%)
        
        Returns:
            Dictionary containing warnings and current resource usage
        """
        warnings = []
        resource_metrics = self.get_resource_metrics()
        
        # Check CPU usage
        cpu_percent = resource_metrics["cpu_percent"]
        if cpu_percent >= cpu_threshold:
            warning_msg = f"CPU usage at {cpu_percent:.1f}% (threshold: {cpu_threshold}%)"
            warnings.append(warning_msg)
            logger.warning(
                "Resource limit warning: High CPU usage",
                extra={
                    "cpu_percent": cpu_percent,
                    "threshold": cpu_threshold,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Check memory usage
        memory_percent = resource_metrics["memory"]["percent"]
        if memory_percent >= memory_threshold:
            warning_msg = f"Memory usage at {memory_percent:.1f}% (threshold: {memory_threshold}%)"
            warnings.append(warning_msg)
            logger.warning(
                "Resource limit warning: High memory usage",
                extra={
                    "memory_percent": memory_percent,
                    "memory_used_mb": resource_metrics["memory"]["used_mb"],
                    "memory_total_mb": resource_metrics["memory"]["total_mb"],
                    "threshold": memory_threshold,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Check GPU memory if available
        if resource_metrics["gpu"]:
            for gpu in resource_metrics["gpu"]["devices"]:
                # Calculate GPU memory usage percentage
                allocated = gpu["memory_allocated_mb"]
                reserved = gpu["memory_reserved_mb"]
                if reserved > 0:
                    gpu_percent = (allocated / reserved) * 100
                    if gpu_percent >= memory_threshold:
                        warning_msg = f"GPU {gpu['device_id']} memory at {gpu_percent:.1f}% (threshold: {memory_threshold}%)"
                        warnings.append(warning_msg)
                        logger.warning(
                            "Resource limit warning: High GPU memory usage",
                            extra={
                                "gpu_device_id": gpu["device_id"],
                                "gpu_memory_percent": gpu_percent,
                                "gpu_memory_allocated_mb": allocated,
                                "gpu_memory_reserved_mb": reserved,
                                "threshold": memory_threshold,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
        
        return {
            "warnings": warnings,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "cpu_threshold": cpu_threshold,
            "memory_threshold": memory_threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        model_status = self.get_model_status()
        cache_status = self.check_cache_status()
        api_status = self.check_trendyol_api_status()
        
        # Determine overall status
        if model_status["loaded"] and cache_status == "healthy" and api_status == "healthy":
            overall_status = "healthy"
        elif model_status["loaded"]:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "model_loaded": model_status["loaded"],
            "model_load_time": model_status["load_time"],
            "trendyol_api_status": api_status,
            "cache_status": cache_status,
            "timestamp": datetime.utcnow()
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        metrics = []
        
        # Add HELP and TYPE comments for better Prometheus integration
        metrics.append('# HELP container_cpu_usage_percent Container CPU usage percentage')
        metrics.append('# TYPE container_cpu_usage_percent gauge')
        
        metrics.append('# HELP container_memory_usage_percent Container memory usage percentage')
        metrics.append('# TYPE container_memory_usage_percent gauge')
        
        metrics.append('# HELP container_memory_used_mb Container memory used in megabytes')
        metrics.append('# TYPE container_memory_used_mb gauge')
        
        metrics.append('# HELP container_uptime_seconds Container uptime in seconds')
        metrics.append('# TYPE container_uptime_seconds counter')
        
        metrics.append('# HELP container_restart_count Total container restart count')
        metrics.append('# TYPE container_restart_count counter')
        
        # Tool usage metrics
        tool_stats = self.metrics_collector.get_tool_stats()
        for tool_name, count in tool_stats["tool_usage"].items():
            metrics.append(f'tool_usage{{tool="{tool_name}"}} {count}')
        
        for tool_name, rate in tool_stats["success_rates"].items():
            metrics.append(f'tool_success_rate{{tool="{tool_name}"}} {rate}')
        
        for tool_name, time in tool_stats["average_execution_times"].items():
            metrics.append(f'tool_execution_time_seconds{{tool="{tool_name}"}} {time}')
        
        # Performance metrics
        perf_metrics = self.metrics_collector.get_performance_metrics()
        metrics.append(f'inference_time_seconds {perf_metrics["average_inference_time"]}')
        metrics.append(f'api_response_time_seconds {perf_metrics["average_response_time"]}')
        metrics.append(f'total_inferences {perf_metrics["total_inferences"]}')
        metrics.append(f'total_requests {perf_metrics["total_requests"]}')
        
        # Resource metrics (Container-specific)
        resource_metrics = self.get_resource_metrics()
        metrics.append(f'container_cpu_usage_percent {resource_metrics["cpu_percent"]}')
        metrics.append(f'container_memory_usage_percent {resource_metrics["memory"]["percent"]}')
        metrics.append(f'container_memory_used_mb {resource_metrics["memory"]["used_mb"]}')
        metrics.append(f'container_memory_total_mb {resource_metrics["memory"]["total_mb"]}')
        metrics.append(f'container_memory_available_mb {resource_metrics["memory"]["available_mb"]}')
        
        # Container lifecycle metrics
        uptime_seconds = (datetime.utcnow() - self._container_start_time).total_seconds()
        metrics.append(f'container_uptime_seconds {uptime_seconds}')
        metrics.append(f'container_restart_count {self._restart_count}')
        
        # Health check failure metrics
        for check_name, failure_count in self._health_check_failures.items():
            metrics.append(f'health_check_failures{{check="{check_name}"}} {failure_count}')
        
        if resource_metrics["gpu"]:
            for gpu in resource_metrics["gpu"]["devices"]:
                device_id = gpu["device_id"]
                metrics.append(
                    f'gpu_memory_allocated_mb{{device="{device_id}"}} '
                    f'{gpu["memory_allocated_mb"]}'
                )
        
        # Model status
        metrics.append(f'model_loaded {1 if self._model_loaded else 0}')
        
        return "\n".join(metrics) + "\n"


# Global monitoring service instance
monitoring_service = MonitoringService()
