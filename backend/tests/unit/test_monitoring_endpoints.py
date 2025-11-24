"""Unit tests for monitoring endpoints"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from app.services.monitoring_service import monitoring_service


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_returns_200(self, client):
        """Test that health check returns 200 status"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
    
    def test_health_check_response_structure(self, client):
        """Test health check response has correct structure"""
        response = client.get("/api/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "trendyol_api_status" in data
        assert "cache_status" in data
        assert "timestamp" in data
    
    def test_health_check_status_values(self, client):
        """Test health check status has valid values"""
        response = client.get("/api/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)
        assert data["trendyol_api_status"] in ["healthy", "unhealthy", "unknown"]
        assert data["cache_status"] in ["healthy", "unhealthy", "unknown"]
    
    def test_health_check_timestamp_format(self, client):
        """Test health check timestamp is valid ISO format"""
        response = client.get("/api/health")
        data = response.json()
        
        # Should be able to parse as datetime
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""
    
    def test_metrics_returns_200(self, client):
        """Test that metrics endpoint returns 200 status"""
        response = client.get("/api/metrics")
        
        assert response.status_code == 200
    
    def test_metrics_content_type(self, client):
        """Test metrics endpoint returns correct content type"""
        response = client.get("/api/metrics")
        
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_format(self, client):
        """Test metrics are in Prometheus format"""
        response = client.get("/api/metrics")
        content = response.text
        
        # Should contain metric names
        assert "cpu_usage_percent" in content
        assert "memory_usage_percent" in content
        assert "model_loaded" in content
    
    def test_metrics_with_tool_data(self, client):
        """Test metrics include tool data when available"""
        # Record some tool execution
        monitoring_service.metrics_collector.record_tool_execution(
            "test_tool", 1.5, True
        )
        
        response = client.get("/api/metrics")
        content = response.text
        
        assert "tool_usage" in content
        assert "test_tool" in content


class TestResourcesEndpoint:
    """Test resources endpoint"""
    
    def test_resources_returns_200(self, client):
        """Test that resources endpoint returns 200 status"""
        response = client.get("/api/resources")
        
        assert response.status_code == 200
    
    def test_resources_response_structure(self, client):
        """Test resources response has correct structure"""
        response = client.get("/api/resources")
        data = response.json()
        
        assert "cpu_percent" in data
        assert "memory" in data
        assert "timestamp" in data
    
    def test_resources_memory_structure(self, client):
        """Test memory information has correct structure"""
        response = client.get("/api/resources")
        data = response.json()
        
        memory = data["memory"]
        assert "total_mb" in memory
        assert "used_mb" in memory
        assert "available_mb" in memory
        assert "percent" in memory
    
    def test_resources_values_valid(self, client):
        """Test resource values are valid"""
        response = client.get("/api/resources")
        data = response.json()
        
        assert 0 <= data["cpu_percent"] <= 100
        assert 0 <= data["memory"]["percent"] <= 100
        assert data["memory"]["total_mb"] > 0


class TestToolStatsEndpoint:
    """Test tool statistics endpoint"""
    
    def test_tool_stats_returns_200(self, client):
        """Test that tool stats endpoint returns 200 status"""
        response = client.get("/api/tools/stats")
        
        assert response.status_code == 200
    
    def test_tool_stats_response_structure(self, client):
        """Test tool stats response has correct structure"""
        response = client.get("/api/tools/stats")
        data = response.json()
        
        assert "tool_usage" in data
        assert "success_rates" in data
        assert "average_execution_times" in data
    
    def test_tool_stats_with_data(self, client):
        """Test tool stats with recorded data"""
        # Record some tool executions
        monitoring_service.metrics_collector.record_tool_execution(
            "price_comparison", 1.0, True
        )
        monitoring_service.metrics_collector.record_tool_execution(
            "price_comparison", 2.0, True
        )
        monitoring_service.metrics_collector.record_tool_execution(
            "review_analysis", 1.5, False
        )
        
        response = client.get("/api/tools/stats")
        data = response.json()
        
        assert "price_comparison" in data["tool_usage"]
        assert "review_analysis" in data["tool_usage"]
        assert data["tool_usage"]["price_comparison"] >= 2
        assert data["success_rates"]["price_comparison"] > 0


class TestPerformanceEndpoint:
    """Test performance metrics endpoint"""
    
    def test_performance_returns_200(self, client):
        """Test that performance endpoint returns 200 status"""
        response = client.get("/api/performance")
        
        assert response.status_code == 200
    
    def test_performance_response_structure(self, client):
        """Test performance response has correct structure"""
        response = client.get("/api/performance")
        data = response.json()
        
        assert "average_inference_time" in data
        assert "average_response_time" in data
        assert "total_inferences" in data
        assert "total_requests" in data
    
    def test_performance_with_data(self, client):
        """Test performance metrics with recorded data"""
        # Record some metrics
        monitoring_service.metrics_collector.record_inference_time(2.5)
        monitoring_service.metrics_collector.record_api_response_time(0.5)
        
        response = client.get("/api/performance")
        data = response.json()
        
        assert data["total_inferences"] >= 1
        assert data["total_requests"] >= 1
        assert data["average_inference_time"] > 0
        assert data["average_response_time"] > 0
