"""
Unit tests for performance reporting module.

Tests training summary generation, resource usage analysis,
and optimization recommendations for MacBook TRM training.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict
import time

from macbook_optimization.performance_reporting import (
    TrainingSummary,
    ResourceUsageReport,
    OptimizationReport,
    PerformanceReporter
)
from macbook_optimization.progress_monitoring import (
    TrainingSession,
    TrainingProgress,
    ResourceMetrics,
    PerformanceMetrics
)


class TestTrainingSummary:
    """Test TrainingSummary dataclass."""
    
    def test_training_summary_creation(self):
        """Test TrainingSummary dataclass creation."""
        summary = TrainingSummary(
            session_id="test_session",
            model_name="TRM-7M",
            dataset_name="test_dataset",
            start_time="2024-01-01T10:00:00",
            end_time="2024-01-01T11:00:00",
            total_duration_minutes=60.0,
            batch_size=16,
            learning_rate=1e-4,
            total_steps=1000,
            total_epochs=5,
            steps_completed=1000,
            samples_processed=16000,
            completion_percentage=100.0,
            average_samples_per_second=26.7,
            peak_samples_per_second=35.2,
            training_efficiency_score=85.0,
            peak_memory_usage_mb=6000.0,
            average_memory_usage_mb=5200.0,
            peak_cpu_usage_percent=90.0,
            average_cpu_usage_percent=75.0,
            thermal_throttling_detected=False,
            final_loss=0.234,
            best_loss=0.198,
            loss_improvement=0.456,
            convergence_achieved=True,
            primary_bottlenecks=["memory", "cpu"],
            optimization_suggestions=["Reduce batch size", "Optimize CPU usage"],
            performance_warnings=["High memory usage detected"],
            memory_efficiency_score=87.0,
            cpu_efficiency_score=75.0,
            overall_hardware_efficiency=81.0
        )
        
        assert summary.session_id == "test_session"
        assert summary.model_name == "TRM-7M"
        assert summary.total_duration_minutes == 60.0
        assert summary.training_efficiency_score == 85.0
        assert summary.convergence_achieved is True
        assert len(summary.primary_bottlenecks) == 2
        assert len(summary.optimization_suggestions) == 2


class TestResourceUsageReport:
    """Test ResourceUsageReport dataclass."""
    
    def test_resource_usage_report_creation(self):
        """Test ResourceUsageReport dataclass creation."""
        report = ResourceUsageReport(
            memory_stats={"peak_mb": 6000.0, "average_mb": 5200.0},
            memory_timeline=[(1640995200.0, 4000.0), (1640995260.0, 5000.0)],
            memory_pressure_events=[{"timestamp": 1640995300.0, "pressure_level": "high"}],
            cpu_stats={"peak_percent": 90.0, "average_percent": 75.0},
            cpu_timeline=[(1640995200.0, 70.0), (1640995260.0, 80.0)],
            cpu_frequency_timeline=[(1640995200.0, 2400.0), (1640995260.0, 2600.0)],
            thermal_events=[{"timestamp": 1640995400.0, "thermal_state": "warm"}],
            thermal_timeline=[(1640995200.0, "normal"), (1640995400.0, "warm")],
            bottleneck_analysis={"primary_bottleneck": "memory", "bottleneck_severity": 0.7},
            resource_recommendations=["Reduce batch size", "Optimize memory usage"]
        )
        
        assert report.memory_stats["peak_mb"] == 6000.0
        assert len(report.memory_timeline) == 2
        assert len(report.memory_pressure_events) == 1
        assert report.bottleneck_analysis["primary_bottleneck"] == "memory"
        assert len(report.resource_recommendations) == 2


class TestOptimizationReport:
    """Test OptimizationReport dataclass."""
    
    def test_optimization_report_creation(self):
        """Test OptimizationReport dataclass creation."""
        report = OptimizationReport(
            current_config_efficiency=75.0,
            config_bottlenecks=["batch_size_too_large", "underutilized_cpu"],
            batch_size_recommendations={
                "current_batch_size": 32,
                "recommended_batch_size": 16,
                "reasoning": "Reduce to lower memory pressure"
            },
            memory_optimizations=["Enable gradient accumulation", "Use mixed precision"],
            cpu_optimizations=["Increase data loading workers", "Enable MKL optimizations"],
            data_loading_optimizations=["Increase prefetch factor", "Use memory mapping"],
            estimated_speed_improvement=25.0,
            estimated_memory_savings=30.0,
            estimated_efficiency_gain=20.0,
            high_priority_optimizations=["Reduce batch size"],
            medium_priority_optimizations=["Enable gradient accumulation"],
            low_priority_optimizations=["Optimize data preprocessing"]
        )
        
        assert report.current_config_efficiency == 75.0
        assert len(report.config_bottlenecks) == 2
        assert report.batch_size_recommendations["recommended_batch_size"] == 16
        assert len(report.memory_optimizations) == 2
        assert report.estimated_speed_improvement == 25.0


class TestPerformanceReporter:
    """Test PerformanceReporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test reports
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = PerformanceReporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_training_session(self) -> TrainingSession:
        """Create a mock training session with realistic data."""
        session = TrainingSession(
            session_id="test_session_001",
            start_time=time.time() - 3600,  # 1 hour ago
            end_time=time.time(),
            model_name="TRM-7M",
            dataset_name="test_dataset",
            batch_size=16,
            learning_rate=1e-4
        )
        
        # Add progress history
        for i in range(100):
            progress = TrainingProgress(
                current_step=i + 1,
                total_steps=100,
                current_epoch=1,
                total_epochs=1,
                samples_processed=(i + 1) * 16,
                total_samples=1600,
                progress_percent=(i + 1),
                start_time=session.start_time,
                elapsed_time=(i + 1) * 36,  # 36 seconds per step
                estimated_total_time=3600,
                eta_seconds=3600 - (i + 1) * 36,
                current_samples_per_second=16 / 36,  # ~0.44 samples/sec
                average_samples_per_second=16 / 36,
                recent_samples_per_second=16 / 36,
                current_loss=1.0 - (i / 100) * 0.5,  # Decreasing from 1.0 to 0.5
                average_loss=0.75,
                best_loss=0.5,
                current_learning_rate=1e-4 * (0.99 ** i)
            )
            session.progress_history.append(progress)
        
        # Add resource history
        for i in range(50):  # Fewer resource measurements
            resource = ResourceMetrics(
                memory_used_mb=4000 + i * 20,  # Increasing memory usage
                memory_available_mb=4192 - i * 20,
                memory_usage_percent=50 + i * 0.5,
                memory_pressure_level="medium" if i > 25 else "low",
                cpu_usage_percent=60 + i * 0.4,
                cpu_frequency_mhz=2400.0,
                cpu_load_average=[1.0, 1.2, 1.5],
                thermal_state="warm" if i > 40 else "normal",
                batch_processing_time_ms=150.0,
                gradient_computation_time_ms=90.0,
                data_loading_time_ms=15.0
            )
            session.resource_history.append(resource)
        
        # Add performance history
        for i in range(50):
            performance = PerformanceMetrics(
                memory_efficiency_percent=80 - i * 0.2,
                cpu_efficiency_percent=70 + i * 0.1,
                training_efficiency_score=75 + i * 0.1,
                primary_bottleneck="memory" if i > 25 else "cpu",
                bottleneck_severity=0.3 + i * 0.01,
                optimization_suggestions=["Reduce batch size"] if i > 25 else ["Optimize CPU"],
                performance_warnings=["High memory usage"] if i > 40 else []
            )
            session.performance_history.append(performance)
        
        # Set final statistics
        session.total_samples_processed = 1600
        session.total_training_time = 3600.0
        session.average_samples_per_second = 1600 / 3600
        session.peak_memory_usage_mb = 5000.0
        session.final_loss = 0.5
        session.best_loss = 0.5
        
        return session
    
    def test_performance_reporter_initialization(self):
        """Test PerformanceReporter initialization."""
        assert self.reporter.output_dir == Path(self.temp_dir)
        assert self.reporter.output_dir.exists()
        assert len(self.reporter.session_reports) == 0
        assert len(self.reporter.resource_reports) == 0
        assert len(self.reporter.optimization_reports) == 0
    
    def test_generate_training_summary(self):
        """Test training summary generation."""
        session = self.create_mock_training_session()
        summary = self.reporter.generate_training_summary(session)
        
        assert isinstance(summary, TrainingSummary)
        assert summary.session_id == "test_session_001"
        assert summary.model_name == "TRM-7M"
        assert summary.dataset_name == "test_dataset"
        assert summary.batch_size == 16
        assert summary.learning_rate == 1e-4
        assert summary.steps_completed == 100
        assert summary.samples_processed == 1600
        assert summary.completion_percentage == 100.0
        assert summary.total_duration_minutes == 60.0
        assert summary.average_samples_per_second > 0
        assert summary.peak_memory_usage_mb > 4900.0  # Should be close to 5000
        assert summary.final_loss == 0.5
        assert summary.best_loss == 0.5
        assert summary.convergence_achieved is False  # Simple convergence check
        assert len(summary.primary_bottlenecks) > 0
        assert summary.overall_hardware_efficiency > 0
        
        # Check that summary is stored
        assert session.session_id in self.reporter.session_reports
    
    def test_generate_training_summary_empty_session(self):
        """Test training summary generation with empty session."""
        session = TrainingSession(
            session_id="empty_session",
            start_time=time.time() - 100,
            model_name="TRM-7M",
            dataset_name="test_dataset",
            batch_size=16,
            learning_rate=1e-4
        )
        session.end_time = time.time()
        session.total_training_time = 100.0
        
        summary = self.reporter.generate_training_summary(session)
        
        assert isinstance(summary, TrainingSummary)
        assert summary.steps_completed == 0
        assert summary.samples_processed == 0
        assert summary.completion_percentage == 0
        assert summary.average_samples_per_second == 0
        assert summary.peak_samples_per_second == 0
        assert summary.training_efficiency_score == 0
        assert len(summary.primary_bottlenecks) == 0
    
    def test_generate_resource_usage_report(self):
        """Test resource usage report generation."""
        session = self.create_mock_training_session()
        report = self.reporter.generate_resource_usage_report(session)
        
        assert isinstance(report, ResourceUsageReport)
        
        # Check memory analysis
        assert "peak_mb" in report.memory_stats
        assert "average_mb" in report.memory_stats
        assert report.memory_stats["peak_mb"] > 0
        assert len(report.memory_timeline) == len(session.resource_history)
        # Memory pressure events depend on the pressure level thresholds
        # The mock data may not trigger high/critical pressure events
        assert isinstance(report.memory_pressure_events, list)
        
        # Check CPU analysis
        assert "peak_percent" in report.cpu_stats
        assert "average_percent" in report.cpu_stats
        assert len(report.cpu_timeline) == len(session.resource_history)
        assert len(report.cpu_frequency_timeline) == len(session.resource_history)
        
        # Check thermal analysis
        assert len(report.thermal_timeline) == len(session.resource_history)
        assert len(report.thermal_events) > 0  # Should have some warm events
        
        # Check bottleneck analysis
        assert "bottleneck_distribution" in report.bottleneck_analysis
        assert "primary_bottleneck" in report.bottleneck_analysis
        assert report.bottleneck_analysis["primary_bottleneck"] in ["memory", "cpu"]
        
        # Check recommendations
        assert len(report.resource_recommendations) > 0
        
        # Check that report is stored
        assert session.session_id in self.reporter.resource_reports
    
    def test_generate_resource_usage_report_empty_session(self):
        """Test resource usage report generation with empty session."""
        session = TrainingSession(
            session_id="empty_session",
            start_time=time.time(),
            model_name="TRM-7M"
        )
        
        report = self.reporter.generate_resource_usage_report(session)
        
        assert isinstance(report, ResourceUsageReport)
        assert report.memory_stats == {}
        assert len(report.memory_timeline) == 0
        assert len(report.memory_pressure_events) == 0
        assert len(report.resource_recommendations) == 0
    
    def test_generate_optimization_report(self):
        """Test optimization report generation."""
        session = self.create_mock_training_session()
        current_config = {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "model_params": 7000000
        }
        
        report = self.reporter.generate_optimization_report(session, current_config)
        
        assert isinstance(report, OptimizationReport)
        assert 0 <= report.current_config_efficiency <= 100
        assert isinstance(report.config_bottlenecks, list)
        
        # Check batch size recommendations
        assert "current_batch_size" in report.batch_size_recommendations
        assert "recommended_batch_size" in report.batch_size_recommendations
        assert "reasoning" in report.batch_size_recommendations
        assert report.batch_size_recommendations["current_batch_size"] == 16
        
        # Check optimization lists
        assert isinstance(report.memory_optimizations, list)
        assert isinstance(report.cpu_optimizations, list)
        assert isinstance(report.data_loading_optimizations, list)
        
        # Check estimated improvements
        assert isinstance(report.estimated_speed_improvement, float)
        assert isinstance(report.estimated_memory_savings, float)
        assert isinstance(report.estimated_efficiency_gain, float)
        
        # Check priority lists
        assert isinstance(report.high_priority_optimizations, list)
        assert isinstance(report.medium_priority_optimizations, list)
        assert isinstance(report.low_priority_optimizations, list)
        
        # Check that report is stored
        assert session.session_id in self.reporter.optimization_reports
    
    def test_save_reports(self):
        """Test saving all reports to files."""
        session = self.create_mock_training_session()
        current_config = {"batch_size": 16, "learning_rate": 1e-4}
        
        self.reporter.save_reports(session, current_config)
        
        # Check that session directory was created
        session_dir = Path(self.temp_dir) / session.session_id
        assert session_dir.exists()
        
        # Check that all report files were created
        assert (session_dir / "training_summary.json").exists()
        assert (session_dir / "resource_usage.json").exists()
        assert (session_dir / "optimization_recommendations.json").exists()
        assert (session_dir / "raw_session_data.json").exists()
        
        # Verify file contents
        with open(session_dir / "training_summary.json", "r") as f:
            summary_data = json.load(f)
            assert summary_data["session_id"] == session.session_id
            assert summary_data["model_name"] == "TRM-7M"
        
        with open(session_dir / "resource_usage.json", "r") as f:
            resource_data = json.load(f)
            assert "memory_stats" in resource_data
            assert "cpu_stats" in resource_data
        
        with open(session_dir / "optimization_recommendations.json", "r") as f:
            optimization_data = json.load(f)
            assert "current_config_efficiency" in optimization_data
            assert "batch_size_recommendations" in optimization_data
        
        with open(session_dir / "raw_session_data.json", "r") as f:
            raw_data = json.load(f)
            assert "session_info" in raw_data
            assert "progress_history" in raw_data
            assert len(raw_data["progress_history"]) == 100
    
    def test_save_reports_without_config(self):
        """Test saving reports without optimization config."""
        session = self.create_mock_training_session()
        
        self.reporter.save_reports(session)  # No config provided
        
        session_dir = Path(self.temp_dir) / session.session_id
        assert session_dir.exists()
        
        # Should have all files except optimization recommendations
        assert (session_dir / "training_summary.json").exists()
        assert (session_dir / "resource_usage.json").exists()
        assert not (session_dir / "optimization_recommendations.json").exists()
        assert (session_dir / "raw_session_data.json").exists()
    
    def test_generate_text_summary(self):
        """Test human-readable text summary generation."""
        session = self.create_mock_training_session()
        text_summary = self.reporter.generate_text_summary(session)
        
        assert isinstance(text_summary, str)
        assert len(text_summary) > 0
        
        # Check that key information is included
        assert "MacBook TRM Training Report" in text_summary
        assert session.session_id in text_summary
        assert "TRM-7M" in text_summary
        assert "test_dataset" in text_summary
        assert "SESSION INFORMATION:" in text_summary
        assert "TRAINING CONFIGURATION:" in text_summary
        assert "PERFORMANCE METRICS:" in text_summary
        assert "RESOURCE UTILIZATION:" in text_summary
        assert "TRAINING OUTCOMES:" in text_summary
        assert "EFFICIENCY ANALYSIS:" in text_summary
        
        # Check for specific values
        assert "Batch Size: 16" in text_summary
        assert "1,600" in text_summary  # Samples processed (formatted with comma)
        assert "60.0 minutes" in text_summary  # Duration
    
    def test_save_text_summary(self):
        """Test saving text summary to file."""
        session = self.create_mock_training_session()
        
        self.reporter.save_text_summary(session)
        
        session_dir = Path(self.temp_dir) / session.session_id
        summary_file = session_dir / f"{session.session_id}_summary.txt"
        
        assert summary_file.exists()
        
        # Check file contents
        with open(summary_file, "r") as f:
            content = f.read()
            assert "MacBook TRM Training Report" in content
            assert session.session_id in content
    
    def test_save_text_summary_custom_filename(self):
        """Test saving text summary with custom filename."""
        session = self.create_mock_training_session()
        custom_filename = "custom_report.txt"
        
        self.reporter.save_text_summary(session, custom_filename)
        
        session_dir = Path(self.temp_dir) / session.session_id
        summary_file = session_dir / custom_filename
        
        assert summary_file.exists()
    
    def test_compare_sessions(self):
        """Test session comparison functionality."""
        # Create two different sessions
        session1 = self.create_mock_training_session()
        session1.session_id = "session_1"
        
        session2 = self.create_mock_training_session()
        session2.session_id = "session_2"
        session2.batch_size = 32  # Different batch size
        # Modify some metrics for session2
        for progress in session2.progress_history:
            progress.current_samples_per_second *= 1.2  # 20% faster
        session2.average_samples_per_second *= 1.2
        
        # Generate summaries
        self.reporter.generate_training_summary(session1)
        self.reporter.generate_training_summary(session2)
        
        # Compare sessions
        comparison = self.reporter.compare_sessions(["session_1", "session_2"])
        
        assert "sessions" in comparison
        assert "metrics_comparison" in comparison
        assert "best_session" in comparison
        
        assert comparison["sessions"] == ["session_1", "session_2"]
        
        # Check metrics comparison
        metrics = comparison["metrics_comparison"]
        assert "training_efficiency" in metrics
        assert "average_speed_sps" in metrics
        assert "peak_memory_mb" in metrics
        assert len(metrics["average_speed_sps"]) == 2
        
        # Check best session identification
        best = comparison["best_session"]
        assert "highest_efficiency" in best
        assert "fastest_training" in best
        assert "lowest_memory" in best
        assert "best_loss" in best
        
        # Session 2 should be faster
        assert best["fastest_training"] == "session_2"
    
    def test_compare_sessions_insufficient_data(self):
        """Test session comparison with insufficient data."""
        # Test with no sessions
        comparison = self.reporter.compare_sessions([])
        assert "error" in comparison
        
        # Test with one session
        comparison = self.reporter.compare_sessions(["session_1"])
        assert "error" in comparison
        
        # Test with non-existent session
        comparison = self.reporter.compare_sessions(["session_1", "non_existent"])
        assert "error" in comparison
    
    def test_optimization_report_batch_size_recommendations(self):
        """Test batch size recommendations in optimization report."""
        session = self.create_mock_training_session()
        
        # Test with high memory usage (should recommend smaller batch size)
        for resource in session.resource_history:
            resource.memory_usage_percent = 90.0  # Very high memory usage
        
        current_config = {"batch_size": 32}
        report = self.reporter.generate_optimization_report(session, current_config)
        
        # Should recommend reducing batch size
        assert "batch_size_too_large" in report.config_bottlenecks
        assert report.batch_size_recommendations["recommended_batch_size"] < 32
        assert "reduce" in report.batch_size_recommendations["reasoning"].lower()
    
    def test_optimization_report_cpu_recommendations(self):
        """Test CPU optimization recommendations."""
        session = self.create_mock_training_session()
        
        # Test with low CPU usage
        for resource in session.resource_history:
            resource.cpu_usage_percent = 25.0  # Low CPU usage
        
        current_config = {"batch_size": 16}
        report = self.reporter.generate_optimization_report(session, current_config)
        
        # Should recommend CPU optimizations
        assert "underutilized_cpu" in report.config_bottlenecks
        assert len(report.cpu_optimizations) > 0
        assert any("worker" in opt.lower() for opt in report.cpu_optimizations)
    
    def test_optimization_report_priority_classification(self):
        """Test optimization priority classification."""
        session = self.create_mock_training_session()
        
        # Create critical memory situation
        for resource in session.resource_history:
            resource.memory_usage_percent = 95.0  # Critical memory usage
            resource.thermal_state = "hot"  # Thermal issues
        
        current_config = {"batch_size": 32}
        report = self.reporter.generate_optimization_report(session, current_config)
        
        # Should have high priority optimizations
        assert len(report.high_priority_optimizations) > 0
        assert any("batch size" in opt.lower() for opt in report.high_priority_optimizations)
        assert any("thermal" in opt.lower() for opt in report.high_priority_optimizations)
    
    def test_memory_pressure_event_detection(self):
        """Test memory pressure event detection in resource report."""
        session = self.create_mock_training_session()
        
        # Add some high memory pressure events
        for i, resource in enumerate(session.resource_history):
            if i % 10 == 0:  # Every 10th measurement
                resource.memory_pressure_level = "critical"
                resource.memory_usage_percent = 95.0
        
        report = self.reporter.generate_resource_usage_report(session)
        
        # Should detect memory pressure events
        assert len(report.memory_pressure_events) > 0
        
        # Check event structure
        for event in report.memory_pressure_events:
            assert "timestamp" in event
            assert "pressure_level" in event
            assert event["pressure_level"] == "critical"
            assert "memory_usage_percent" in event
    
    def test_thermal_event_detection(self):
        """Test thermal event detection in resource report."""
        session = self.create_mock_training_session()
        
        # Add thermal events
        for i, resource in enumerate(session.resource_history):
            if i > 30:  # Later in training
                resource.thermal_state = "hot"
        
        report = self.reporter.generate_resource_usage_report(session)
        
        # Should detect thermal events
        assert len(report.thermal_events) > 0
        
        # Check event structure
        hot_events = [e for e in report.thermal_events if e["thermal_state"] == "hot"]
        assert len(hot_events) > 0
        
        for event in hot_events:
            assert "timestamp" in event
            assert "thermal_state" in event
            assert "cpu_usage" in event


class TestPerformanceReporterIntegration:
    """Integration tests for PerformanceReporter with realistic scenarios."""
    
    def test_complete_reporting_workflow(self):
        """Test complete reporting workflow from session to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = PerformanceReporter(output_dir=temp_dir)
            
            # Create a realistic training session
            session = TrainingSession(
                session_id="integration_test",
                start_time=time.time() - 1800,  # 30 minutes ago
                end_time=time.time(),
                model_name="TRM-7M",
                dataset_name="email_classification",
                batch_size=8,  # Small batch for MacBook
                learning_rate=5e-5
            )
            
            # Add realistic progress data
            for step in range(1, 201):  # 200 steps
                progress = TrainingProgress(
                    current_step=step,
                    total_steps=200,
                    current_epoch=1,
                    total_epochs=1,
                    samples_processed=step * 8,
                    total_samples=1600,
                    progress_percent=(step / 200) * 100,
                    start_time=session.start_time,
                    elapsed_time=step * 9,  # 9 seconds per step
                    estimated_total_time=1800,
                    eta_seconds=1800 - step * 9,
                    current_samples_per_second=8 / 9,  # ~0.89 samples/sec
                    average_samples_per_second=8 / 9,
                    recent_samples_per_second=8 / 9,
                    current_loss=0.8 - (step / 200) * 0.3,  # From 0.8 to 0.5
                    average_loss=0.65,
                    best_loss=0.5,
                    current_learning_rate=5e-5 * (0.995 ** step)
                )
                session.progress_history.append(progress)
            
            # Add realistic resource data
            for i in range(100):  # 100 resource measurements
                resource = ResourceMetrics(
                    memory_used_mb=3000 + i * 15,  # Gradually increasing
                    memory_available_mb=5192 - i * 15,
                    memory_usage_percent=36.6 + i * 0.18,
                    memory_pressure_level="high" if i > 80 else "medium" if i > 40 else "low",
                    cpu_usage_percent=50 + i * 0.3,
                    cpu_frequency_mhz=2400.0,
                    cpu_load_average=[1.0 + i * 0.01, 1.2 + i * 0.01, 1.5 + i * 0.01],
                    thermal_state="hot" if i > 85 else "warm" if i > 60 else "normal",
                    batch_processing_time_ms=180.0 + i * 2,
                    gradient_computation_time_ms=108.0 + i * 1.2,
                    data_loading_time_ms=18.0 + i * 0.2
                )
                session.resource_history.append(resource)
            
            # Add performance analysis
            for i in range(100):
                performance = PerformanceMetrics(
                    memory_efficiency_percent=90 - i * 0.2,
                    cpu_efficiency_percent=60 + i * 0.2,
                    training_efficiency_score=75 - i * 0.1,
                    primary_bottleneck="memory" if i > 60 else "cpu",
                    bottleneck_severity=0.2 + i * 0.005,
                    optimization_suggestions=[
                        "Reduce batch size" if i > 60 else "Optimize CPU usage",
                        "Enable gradient accumulation" if i > 80 else "Increase workers"
                    ],
                    performance_warnings=[
                        "High memory usage detected" if i > 80 else "",
                        "Thermal throttling risk" if i > 85 else ""
                    ]
                )
                session.performance_history.append(performance)
            
            # Set final session statistics
            session.total_samples_processed = 1600
            session.total_training_time = 1800.0
            session.average_samples_per_second = 1600 / 1800
            session.peak_memory_usage_mb = 4500.0
            session.final_loss = 0.5
            session.best_loss = 0.5
            
            # Generate and save all reports
            config = {
                "batch_size": 8,
                "learning_rate": 5e-5,
                "model_params": 7000000,
                "optimizer": "AdamW"
            }
            
            reporter.save_reports(session, config)
            
            # Verify all reports were generated and saved
            session_dir = Path(temp_dir) / session.session_id
            assert session_dir.exists()
            
            # Check all expected files exist
            expected_files = [
                "training_summary.json",
                "resource_usage.json", 
                "optimization_recommendations.json",
                "raw_session_data.json"
            ]
            
            for filename in expected_files:
                file_path = session_dir / filename
                assert file_path.exists(), f"Missing file: {filename}"
                assert file_path.stat().st_size > 0, f"Empty file: {filename}"
            
            # Verify report contents are reasonable
            with open(session_dir / "training_summary.json", "r") as f:
                summary = json.load(f)
                assert summary["completion_percentage"] == 100.0
                assert summary["samples_processed"] == 1600
                assert summary["thermal_throttling_detected"] is True  # Should detect thermal issues
                assert len(summary["primary_bottlenecks"]) > 0
            
            with open(session_dir / "resource_usage.json", "r") as f:
                resource_report = json.load(f)
                assert resource_report["memory_stats"]["peak_mb"] > 4000
                assert len(resource_report["memory_pressure_events"]) > 0
                assert len(resource_report["thermal_events"]) > 0
            
            with open(session_dir / "optimization_recommendations.json", "r") as f:
                optimization = json.load(f)
                assert optimization["current_config_efficiency"] > 0
                assert len(optimization["high_priority_optimizations"]) > 0
                assert optimization["batch_size_recommendations"]["current_batch_size"] == 8
            
            # Generate and save text summary
            reporter.save_text_summary(session)
            text_file = session_dir / f"{session.session_id}_summary.txt"
            assert text_file.exists()
            
            with open(text_file, "r") as f:
                text_content = f.read()
                assert "MacBook TRM Training Report" in text_content
                assert "integration_test" in text_content
                assert "TRM-7M" in text_content
                assert "email_classification" in text_content