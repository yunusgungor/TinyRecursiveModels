"""Integration tests for monitoring infrastructure"""

import pytest
import time
from app.services.monitoring_service import monitoring_service


class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    def test_metrics_endpoint_exists(self, client):
        """Test that metrics endpoint is accessible"""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_format(self, client):
        """Test that metrics are in Prometheus format"""
        response = client.get("/api/metrics")
        content = response.text
        
        # Check for basic Prometheus metric format
        assert "model_loaded" in content
        assert "cpu_usage_percent" in content
        assert "memory_usage_percent" in content
        
        # Each line should be a metric or comment
        for line in content.split("\n"):
            if line and not line.startswith("#"):
                # Should have format: metric_name{labels} value
                assert " " in line or line == ""
    
    def test_tool_metrics_recorded(self, client):
        """Test that tool execution metrics are recorded"""
        # Record some tool executions
        monitoring_service.metrics_collector.record_tool_execution(
            tool_name="test_tool",
            execution_time=0.5,
            success=True
        )
        monitoring_service.metrics_collector.record_tool_execution(
            tool_name="test_tool",
            execution_time=0.7,
            success=True
        )
        monitoring_service.metrics_collector.record_tool_execution(
            tool_name="test_tool",
            execution_time=1.2,
            success=False
        )
        
        # Get metrics
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that tool metrics are present
        assert 'tool_usage{tool="test_tool"}' in content
        assert 'tool_success_rate{tool="test_tool"}' in content
        assert 'tool_execution_time_seconds{tool="test_tool"}' in content
    
    def test_inference_time_metrics(self, client):
        """Test that inference time metrics are recorded"""
        # Record some inference times
        monitoring_service.metrics_collector.record_inference_time(1.5)
        monitoring_service.metrics_collector.record_inference_time(2.0)
        monitoring_service.metrics_collector.record_inference_time(1.8)
        
        # Get metrics
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that inference metrics are present
        assert "inference_time_seconds" in content
        assert "total_inferences" in content
    
    def test_api_response_time_metrics(self, client):
        """Test that API response time metrics are recorded"""
        # Record some response times
        monitoring_service.metrics_collector.record_api_response_time(0.1)
        monitoring_service.metrics_collector.record_api_response_time(0.2)
        monitoring_service.metrics_collector.record_api_response_time(0.15)
        
        # Get metrics
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that response time metrics are present
        assert "api_response_time_seconds" in content
        assert "total_requests" in content
    
    def test_resource_metrics(self, client):
        """Test that system resource metrics are collected"""
        response = client.get("/api/metrics")
        content = response.text
        
        # Check for resource metrics
        assert "cpu_usage_percent" in content
        assert "memory_usage_percent" in content
        assert "memory_used_mb" in content
        
        # Parse CPU usage value
        for line in content.split("\n"):
            if line.startswith("cpu_usage_percent"):
                value = float(line.split()[-1])
                assert 0 <= value <= 100
                break
    
    def test_tool_stats_endpoint(self, client):
        """Test tool stats endpoint"""
        # Record some tool executions
        monitoring_service.metrics_collector.record_tool_execution(
            tool_name="price_comparison",
            execution_time=0.5,
            success=True
        )
        
        response = client.get("/api/tools/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "tool_usage" in data
        assert "success_rates" in data
        assert "average_execution_times" in data


class TestHealthCheck:
    """Test health check functionality"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "trendyol_api_status" in data
        assert "cache_status" in data
        assert "timestamp" in data
    
    def test_health_status_values(self, client):
        """Test that health status has valid values"""
        response = client.get("/api/health")
        data = response.json()
        
        # Status should be one of: healthy, degraded, unhealthy
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Boolean fields
        assert isinstance(data["model_loaded"], bool)
        
        # Status fields should be strings
        assert isinstance(data["trendyol_api_status"], str)
        assert isinstance(data["cache_status"], str)


class TestAlertTriggers:
    """Test alert trigger conditions"""
    
    def test_high_response_time_detection(self):
        """Test that high response times are detected"""
        # Record high response times
        for _ in range(10):
            monitoring_service.metrics_collector.record_api_response_time(3.0)
        
        # Get performance metrics
        perf_metrics = monitoring_service.metrics_collector.get_performance_metrics()
        avg_response_time = perf_metrics["average_response_time"]
        
        # Should be above alert threshold (2 seconds)
        assert avg_response_time > 2.0
    
    def test_high_inference_time_detection(self):
        """Test that high inference times are detected"""
        # Record high inference times
        for _ in range(10):
            monitoring_service.metrics_collector.record_inference_time(6.0)
        
        # Get performance metrics
        perf_metrics = monitoring_service.metrics_collector.get_performance_metrics()
        avg_inference_time = perf_metrics["average_inference_time"]
        
        # Should be above alert threshold (4.5 seconds to account for previous recordings)
        assert avg_inference_time > 4.5
    
    def test_low_tool_success_rate_detection(self):
        """Test that low tool success rates are detected"""
        # Record mostly failed tool executions
        for i in range(10):
            success = i < 3  # Only 30% success rate
            monitoring_service.metrics_collector.record_tool_execution(
                tool_name="failing_tool",
                execution_time=0.5,
                success=success
            )
        
        # Get tool stats
        tool_stats = monitoring_service.metrics_collector.get_tool_stats()
        success_rate = tool_stats["success_rates"]["failing_tool"]
        
        # Should be below alert threshold (80%)
        assert success_rate < 0.8


class TestDashboardData:
    """Test that dashboard data is available and correct"""
    
    def test_api_performance_dashboard_data(self, client):
        """Test data for API performance dashboard"""
        # Record some data
        monitoring_service.metrics_collector.record_api_response_time(0.5)
        monitoring_service.metrics_collector.record_inference_time(1.5)
        
        # Get metrics
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that all required metrics for dashboard are present
        required_metrics = [
            "api_response_time_seconds",
            "inference_time_seconds",
            "total_requests",
            "total_inferences",
            "model_loaded"
        ]
        
        for metric in required_metrics:
            assert metric in content, f"Missing metric: {metric}"
    
    def test_system_resources_dashboard_data(self, client):
        """Test data for system resources dashboard"""
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that all required metrics for dashboard are present
        required_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "memory_used_mb"
        ]
        
        for metric in required_metrics:
            assert metric in content, f"Missing metric: {metric}"
    
    def test_tool_analytics_dashboard_data(self, client):
        """Test data for tool analytics dashboard"""
        # Record some tool data
        monitoring_service.metrics_collector.record_tool_execution(
            tool_name="test_tool",
            execution_time=0.5,
            success=True
        )
        
        response = client.get("/api/metrics")
        content = response.text
        
        # Check that all required metrics for dashboard are present
        assert "tool_usage" in content
        assert "tool_success_rate" in content
        assert "tool_execution_time_seconds" in content


@pytest.mark.slow
class TestMonitoringPerformance:
    """Test monitoring system performance"""
    
    def test_metrics_collection_performance(self):
        """Test that metrics collection doesn't impact performance"""
        # Measure time to record 1000 metrics
        start_time = time.time()
        
        for i in range(1000):
            monitoring_service.metrics_collector.record_tool_execution(
                tool_name=f"tool_{i % 10}",
                execution_time=0.1,
                success=True
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete in less than 1 second
        assert elapsed_time < 1.0
    
    def test_metrics_endpoint_performance(self, client):
        """Test that metrics endpoint responds quickly"""
        # Record some data
        for i in range(100):
            monitoring_service.metrics_collector.record_tool_execution(
                tool_name=f"tool_{i % 10}",
                execution_time=0.1,
                success=True
            )
        
        # Measure response time
        start_time = time.time()
        response = client.get("/api/metrics")
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        # Should respond in less than 500ms (reasonable for test environment)
        assert elapsed_time < 0.5



