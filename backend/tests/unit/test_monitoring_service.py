"""Unit tests for monitoring service"""

import pytest
from datetime import datetime
from app.services.monitoring_service import MonitoringService, MetricsCollector


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def test_record_tool_execution_success(self):
        """Test recording successful tool execution"""
        collector = MetricsCollector()
        
        collector.record_tool_execution("price_comparison", 1.5, True)
        
        stats = collector.get_tool_stats()
        assert stats["tool_usage"]["price_comparison"] == 1
        assert stats["success_rates"]["price_comparison"] == 1.0
        assert stats["average_execution_times"]["price_comparison"] == 1.5
    
    def test_record_tool_execution_failure(self):
        """Test recording failed tool execution"""
        collector = MetricsCollector()
        
        collector.record_tool_execution("review_analysis", 0.5, False)
        
        stats = collector.get_tool_stats()
        assert stats["tool_usage"]["review_analysis"] == 1
        assert stats["success_rates"]["review_analysis"] == 0.0
    
    def test_record_multiple_tool_executions(self):
        """Test recording multiple tool executions"""
        collector = MetricsCollector()
        
        collector.record_tool_execution("price_comparison", 1.0, True)
        collector.record_tool_execution("price_comparison", 2.0, True)
        collector.record_tool_execution("price_comparison", 1.5, False)
        
        stats = collector.get_tool_stats()
        assert stats["tool_usage"]["price_comparison"] == 3
        assert stats["success_rates"]["price_comparison"] == pytest.approx(2/3, 0.01)
        assert stats["average_execution_times"]["price_comparison"] == pytest.approx(1.5, 0.01)
    
    def test_record_inference_time(self):
        """Test recording inference time"""
        collector = MetricsCollector()
        
        collector.record_inference_time(2.5)
        collector.record_inference_time(3.0)
        
        perf_metrics = collector.get_performance_metrics()
        assert perf_metrics["average_inference_time"] == pytest.approx(2.75, 0.01)
        assert perf_metrics["total_inferences"] == 2
    
    def test_record_api_response_time(self):
        """Test recording API response time"""
        collector = MetricsCollector()
        
        collector.record_api_response_time(0.5)
        collector.record_api_response_time(0.7)
        collector.record_api_response_time(0.6)
        
        perf_metrics = collector.get_performance_metrics()
        assert perf_metrics["average_response_time"] == pytest.approx(0.6, 0.01)
        assert perf_metrics["total_requests"] == 3
    
    def test_metrics_limit(self):
        """Test that metrics are limited to last 1000 entries"""
        collector = MetricsCollector()
        
        # Record 1500 inference times
        for i in range(1500):
            collector.record_inference_time(1.0)
        
        perf_metrics = collector.get_performance_metrics()
        assert perf_metrics["total_inferences"] == 1000


class TestMonitoringService:
    """Test MonitoringService class"""
    
    def test_set_model_loaded(self):
        """Test setting model loaded status"""
        service = MonitoringService()
        
        service.set_model_loaded(True)
        
        status = service.get_model_status()
        assert status["loaded"] is True
        assert status["load_time"] is not None
    
    def test_get_cpu_usage(self):
        """Test getting CPU usage"""
        service = MonitoringService()
        
        cpu_usage = service.get_cpu_usage()
        
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100
    
    def test_get_memory_usage(self):
        """Test getting memory usage"""
        service = MonitoringService()
        
        memory = service.get_memory_usage()
        
        assert "total_mb" in memory
        assert "used_mb" in memory
        assert "available_mb" in memory
        assert "percent" in memory
        assert memory["total_mb"] > 0
        assert 0 <= memory["percent"] <= 100
    
    def test_get_resource_metrics(self):
        """Test getting all resource metrics"""
        service = MonitoringService()
        
        metrics = service.get_resource_metrics()
        
        assert "cpu_percent" in metrics
        assert "memory" in metrics
        assert "gpu" in metrics
        assert "timestamp" in metrics
    
    def test_get_health_status_model_not_loaded(self):
        """Test health status when model is not loaded"""
        service = MonitoringService()
        
        health = service.get_health_status()
        
        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False
    
    def test_get_health_status_model_loaded(self):
        """Test health status when model is loaded"""
        service = MonitoringService()
        service.set_model_loaded(True)
        
        health = service.get_health_status()
        
        assert health["model_loaded"] is True
        assert health["status"] in ["healthy", "degraded"]
    
    def test_export_prometheus_metrics(self):
        """Test exporting metrics in Prometheus format"""
        service = MonitoringService()
        service.set_model_loaded(True)
        service.metrics_collector.record_tool_execution("test_tool", 1.0, True)
        
        metrics_text = service.export_prometheus_metrics()
        
        assert isinstance(metrics_text, str)
        assert "tool_usage" in metrics_text
        assert "model_loaded 1" in metrics_text
        assert "cpu_usage_percent" in metrics_text
