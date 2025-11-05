"""
Hardware Optimization Validation for MacBook Email Training

This module provides comprehensive validation of hardware optimization components
including memory management, CPU optimization, thermal management, and overall
system performance for email classification training.
"""

import time
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import psutil

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    TORCH_AVAILABLE = False

from .macbook_training_pipeline import MacBookTrainingPipeline
from .email_memory_manager import EmailMemoryManager, EmailMemoryConfig
from .memory_management import MemoryConfig
from .email_cpu_optimizer import EmailCPUOptimizer
from .thermal_management import ThermalMonitor
from .email_resource_monitor import EmailResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ValidationTest:
    """Individual validation test result."""
    test_name: str
    category: str  # "memory", "cpu", "thermal", "integration"
    description: str
    passed: bool
    score: float  # 0-100
    execution_time_ms: float
    details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


@dataclass
class ValidationSuite:
    """Complete validation suite result."""
    suite_name: str
    start_time: float
    end_time: float
    total_execution_time: float
    
    # Test results
    tests: List[ValidationTest]
    passed_tests: int
    failed_tests: int
    total_score: float
    
    # Category scores
    memory_score: float
    cpu_score: float
    thermal_score: float
    integration_score: float
    
    # Overall assessment
    overall_grade: str  # "A", "B", "C", "D", "F"
    system_ready: bool
    critical_issues: List[str]
    recommendations: List[str]


class HardwareOptimizationValidator:
    """
    Comprehensive validator for MacBook hardware optimization.
    
    Tests all optimization components under realistic training conditions
    to ensure they work correctly and provide expected performance benefits.
    """
    
    def __init__(self, 
                 training_pipeline: Optional[MacBookTrainingPipeline] = None,
                 test_duration_seconds: float = 60.0):
        """
        Initialize hardware optimization validator.
        
        Args:
            training_pipeline: MacBook training pipeline to validate
            test_duration_seconds: Duration for performance tests
        """
        self.training_pipeline = training_pipeline
        self.test_duration_seconds = test_duration_seconds
        
        # Initialize components for testing
        self.email_memory_manager: Optional[EmailMemoryManager] = None
        self.email_cpu_optimizer: Optional[EmailCPUOptimizer] = None
        self.thermal_monitor: Optional[ThermalMonitor] = None
        self.resource_monitor: Optional[EmailResourceMonitor] = None
        
        # Test results
        self.validation_results: List[ValidationSuite] = []
        
        logger.info("HardwareOptimizationValidator initialized")
    
    def run_comprehensive_validation(self) -> ValidationSuite:
        """
        Run comprehensive hardware optimization validation.
        
        Returns:
            Complete validation suite results
        """
        logger.info("Starting comprehensive hardware optimization validation")
        start_time = time.time()
        
        # Initialize test components
        self._initialize_test_components()
        
        # Run all validation tests
        tests = []
        
        # Memory management tests
        tests.extend(self._run_memory_validation_tests())
        
        # CPU optimization tests
        tests.extend(self._run_cpu_validation_tests())
        
        # Thermal management tests
        tests.extend(self._run_thermal_validation_tests())
        
        # Integration tests
        tests.extend(self._run_integration_validation_tests())
        
        # Calculate results
        end_time = time.time()
        execution_time = end_time - start_time
        
        passed_tests = sum(1 for test in tests if test.passed)
        failed_tests = len(tests) - passed_tests
        
        # Calculate category scores
        memory_tests = [t for t in tests if t.category == "memory"]
        cpu_tests = [t for t in tests if t.category == "cpu"]
        thermal_tests = [t for t in tests if t.category == "thermal"]
        integration_tests = [t for t in tests if t.category == "integration"]
        
        memory_score = sum(t.score for t in memory_tests) / len(memory_tests) if memory_tests else 0
        cpu_score = sum(t.score for t in cpu_tests) / len(cpu_tests) if cpu_tests else 0
        thermal_score = sum(t.score for t in thermal_tests) / len(thermal_tests) if thermal_tests else 0
        integration_score = sum(t.score for t in integration_tests) / len(integration_tests) if integration_tests else 0
        
        total_score = (memory_score + cpu_score + thermal_score + integration_score) / 4
        
        # Determine overall grade
        if total_score >= 90:
            overall_grade = "A"
        elif total_score >= 80:
            overall_grade = "B"
        elif total_score >= 70:
            overall_grade = "C"
        elif total_score >= 60:
            overall_grade = "D"
        else:
            overall_grade = "F"
        
        # Check system readiness
        system_ready = (passed_tests / len(tests) >= 0.8 and 
                       total_score >= 70 and 
                       not any(t.category == "integration" and not t.passed for t in tests))
        
        # Collect critical issues and recommendations
        critical_issues = []
        recommendations = []
        
        for test in tests:
            if not test.passed and test.category == "integration":
                critical_issues.extend(test.errors)
            if test.warnings:
                recommendations.extend(test.warnings)
        
        # Create validation suite result
        validation_suite = ValidationSuite(
            suite_name="MacBook Hardware Optimization Validation",
            start_time=start_time,
            end_time=end_time,
            total_execution_time=execution_time,
            tests=tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_score=total_score,
            memory_score=memory_score,
            cpu_score=cpu_score,
            thermal_score=thermal_score,
            integration_score=integration_score,
            overall_grade=overall_grade,
            system_ready=system_ready,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
        
        self.validation_results.append(validation_suite)
        
        logger.info(f"Hardware optimization validation completed in {execution_time:.1f}s")
        logger.info(f"Overall score: {total_score:.1f}/100 (Grade: {overall_grade})")
        logger.info(f"System ready for training: {system_ready}")
        
        return validation_suite
    
    def _initialize_test_components(self) -> None:
        """Initialize components for testing."""
        logger.info("Initializing test components...")
        
        # Initialize email memory manager
        memory_config = EmailMemoryConfig(
            base_config=MemoryConfig(),
            email_cache_size_mb=50.0,
            model_overhead_multiplier=2.0
        )
        self.email_memory_manager = EmailMemoryManager(memory_config)
        
        # Initialize email CPU optimizer
        self.email_cpu_optimizer = EmailCPUOptimizer(enable_monitoring=True)
        
        # Initialize thermal monitor
        self.thermal_monitor = ThermalMonitor()
        
        # Initialize resource monitor
        self.resource_monitor = EmailResourceMonitor(
            email_memory_manager=self.email_memory_manager
        )
    
    def _run_memory_validation_tests(self) -> List[ValidationTest]:
        """Run memory management validation tests."""
        logger.info("Running memory validation tests...")
        tests = []
        
        # Test 1: Memory detection and configuration
        test_start = time.time()
        try:
            memory_stats = psutil.virtual_memory()
            total_gb = memory_stats.total / (1024**3)
            available_gb = memory_stats.available / (1024**3)
            
            passed = total_gb >= 4 and available_gb >= 2  # Minimum requirements
            score = min(100, (available_gb / 8) * 100)  # Score based on available memory
            
            details = {
                "total_memory_gb": total_gb,
                "available_memory_gb": available_gb,
                "memory_percent_used": memory_stats.percent
            }
            
            warnings = []
            if total_gb < 8:
                warnings.append("Less than 8GB total memory - training may be slow")
            if available_gb < 4:
                warnings.append("Less than 4GB available memory - consider closing other applications")
            
            tests.append(ValidationTest(
                test_name="Memory Detection and Configuration",
                category="memory",
                description="Validate memory detection and basic configuration",
                passed=passed,
                score=score,
                execution_time_ms=(time.time() - test_start) * 1000,
                details=details,
                warnings=warnings,
                errors=[]
            ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Memory Detection and Configuration",
                category="memory",
                description="Validate memory detection and basic configuration",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        # Test 2: Email memory estimation
        test_start = time.time()
        try:
            if self.email_memory_manager:
                memory_breakdown = self.email_memory_manager.estimate_email_model_memory(
                    model_params=1000000,  # 1M parameters
                    batch_size=8,
                    sequence_length=512
                )
                
                total_estimated = memory_breakdown["total"]
                passed = total_estimated > 0 and total_estimated < 8000  # Reasonable range
                score = 100 if passed else 0
                
                details = {
                    "estimated_memory_mb": total_estimated,
                    "memory_breakdown": memory_breakdown
                }
                
                warnings = []
                if total_estimated > 4000:
                    warnings.append("High memory estimation - consider reducing batch size")
                
                tests.append(ValidationTest(
                    test_name="Email Memory Estimation",
                    category="memory",
                    description="Validate email-specific memory estimation",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=warnings,
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Email Memory Estimation",
                category="memory",
                description="Validate email-specific memory estimation",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        # Test 3: Dynamic batch sizing
        test_start = time.time()
        try:
            if self.email_memory_manager:
                batch_recommendation = self.email_memory_manager.calculate_optimal_email_batch_config(
                    model_params=1000000
                )
                
                passed = (batch_recommendation.recommended_batch_size > 0 and 
                         batch_recommendation.recommended_sequence_length > 0)
                score = 100 if passed else 0
                
                details = {
                    "recommended_batch_size": batch_recommendation.recommended_batch_size,
                    "recommended_sequence_length": batch_recommendation.recommended_sequence_length,
                    "estimated_memory_usage_mb": batch_recommendation.estimated_memory_usage_mb
                }
                
                tests.append(ValidationTest(
                    test_name="Dynamic Batch Sizing",
                    category="memory",
                    description="Validate dynamic batch size calculation",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=batch_recommendation.warnings,
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Dynamic Batch Sizing",
                category="memory",
                description="Validate dynamic batch size calculation",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        return tests
    
    def _run_cpu_validation_tests(self) -> List[ValidationTest]:
        """Run CPU optimization validation tests."""
        logger.info("Running CPU validation tests...")
        tests = []
        
        # Test 1: CPU detection and configuration
        test_start = time.time()
        try:
            if self.email_cpu_optimizer:
                cpu_config = self.email_cpu_optimizer.configure_email_cpu_optimization()
                
                passed = (cpu_config.base_config.torch_threads > 0 and
                         cpu_config.email_preprocessing_threads > 0)
                score = 100 if passed else 0
                
                details = {
                    "torch_threads": cpu_config.base_config.torch_threads,
                    "email_preprocessing_threads": cpu_config.email_preprocessing_threads,
                    "dynamic_scaling_enabled": cpu_config.enable_dynamic_scaling,
                    "thermal_management_enabled": cpu_config.enable_thermal_management
                }
                
                warnings = []
                if cpu_config.base_config.torch_threads < 2:
                    warnings.append("Low thread count - performance may be limited")
                
                tests.append(ValidationTest(
                    test_name="CPU Detection and Configuration",
                    category="cpu",
                    description="Validate CPU detection and optimization configuration",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=warnings,
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="CPU Detection and Configuration",
                category="cpu",
                description="Validate CPU detection and optimization configuration",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        # Test 2: CPU performance under load
        test_start = time.time()
        try:
            if self.email_cpu_optimizer and TORCH_AVAILABLE:
                # Start CPU monitoring
                self.email_cpu_optimizer.start_dynamic_scaling()
                
                # Simulate CPU load
                initial_cpu = psutil.cpu_percent(interval=1)
                
                # Create some tensor operations to simulate training load
                for _ in range(10):
                    a = torch.randn(1000, 1000)
                    b = torch.randn(1000, 1000)
                    c = torch.mm(a, b)
                    time.sleep(0.1)
                
                final_cpu = psutil.cpu_percent(interval=1)
                
                # Get CPU metrics
                cpu_metrics = self.email_cpu_optimizer.get_email_cpu_metrics()
                
                passed = (cpu_metrics.cpu_efficiency_score > 50 and
                         not cpu_metrics.thermal_throttling_active)
                score = cpu_metrics.cpu_efficiency_score
                
                details = {
                    "initial_cpu_percent": initial_cpu,
                    "final_cpu_percent": final_cpu,
                    "efficiency_score": cpu_metrics.cpu_efficiency_score,
                    "thermal_throttling": cpu_metrics.thermal_throttling_active,
                    "bottlenecks": cpu_metrics.bottleneck_indicators
                }
                
                self.email_cpu_optimizer.stop_dynamic_scaling()
                
                tests.append(ValidationTest(
                    test_name="CPU Performance Under Load",
                    category="cpu",
                    description="Validate CPU performance under simulated training load",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=cpu_metrics.optimization_opportunities,
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="CPU Performance Under Load",
                category="cpu",
                description="Validate CPU performance under simulated training load",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        return tests
    
    def _run_thermal_validation_tests(self) -> List[ValidationTest]:
        """Run thermal management validation tests."""
        logger.info("Running thermal validation tests...")
        tests = []
        
        # Test 1: Thermal monitoring
        test_start = time.time()
        try:
            if self.thermal_monitor:
                self.thermal_monitor.start_monitoring()
                time.sleep(5)  # Monitor for 5 seconds
                
                thermal_state = self.thermal_monitor.get_thermal_state()
                thermal_summary = self.thermal_monitor.get_thermal_summary()
                
                self.thermal_monitor.stop_monitoring()
                
                passed = thermal_state.thermal_pressure in ["low", "medium", "high", "critical"]
                score = 100 if thermal_state.thermal_pressure in ["low", "medium"] else 50
                
                details = {
                    "thermal_pressure": thermal_state.thermal_pressure,
                    "cpu_temperature": thermal_state.cpu_temperature,
                    "fan_speed": thermal_state.fan_speed,
                    "throttling_active": thermal_state.throttling_active
                }
                
                warnings = []
                if thermal_state.thermal_pressure in ["high", "critical"]:
                    warnings.append("High thermal pressure detected - may affect performance")
                if thermal_state.throttling_active:
                    warnings.append("Thermal throttling active - performance will be reduced")
                
                tests.append(ValidationTest(
                    test_name="Thermal Monitoring",
                    category="thermal",
                    description="Validate thermal monitoring and detection",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=warnings,
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Thermal Monitoring",
                category="thermal",
                description="Validate thermal monitoring and detection",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        return tests
    
    def _run_integration_validation_tests(self) -> List[ValidationTest]:
        """Run integration validation tests."""
        logger.info("Running integration validation tests...")
        tests = []
        
        # Test 1: Complete pipeline initialization
        test_start = time.time()
        try:
            if not self.training_pipeline:
                self.training_pipeline = MacBookTrainingPipeline()
            
            # Test hardware detection
            hardware_specs = self.training_pipeline.detect_and_configure_hardware()
            
            # Test optimization configuration
            training_config = self.training_pipeline.configure_optimization_components()
            
            passed = (hardware_specs is not None and 
                     training_config is not None and
                     self.training_pipeline.is_configured)
            score = 100 if passed else 0
            
            details = {
                "hardware_detected": hardware_specs is not None,
                "optimization_configured": training_config is not None,
                "pipeline_configured": self.training_pipeline.is_configured,
                "memory_tier": hardware_specs.memory_tier if hardware_specs else None,
                "performance_tier": hardware_specs.performance_tier if hardware_specs else None
            }
            
            tests.append(ValidationTest(
                test_name="Complete Pipeline Initialization",
                category="integration",
                description="Validate complete training pipeline initialization",
                passed=passed,
                score=score,
                execution_time_ms=(time.time() - test_start) * 1000,
                details=details,
                warnings=[],
                errors=[]
            ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Complete Pipeline Initialization",
                category="integration",
                description="Validate complete training pipeline initialization",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        # Test 2: Resource monitoring integration
        test_start = time.time()
        try:
            if self.resource_monitor:
                self.resource_monitor.start_training_monitoring("validation_test")
                time.sleep(10)  # Monitor for 10 seconds
                
                dashboard_data = self.resource_monitor.get_real_time_dashboard_data()
                performance_metrics = self.resource_monitor.stop_training_monitoring()
                
                passed = (dashboard_data["status"] == "active" and
                         performance_metrics is not None)
                score = 100 if passed else 0
                
                details = {
                    "monitoring_active": dashboard_data["status"] == "active",
                    "performance_metrics_available": performance_metrics is not None,
                    "dashboard_data_complete": len(dashboard_data.get("current_status", {})) > 0
                }
                
                tests.append(ValidationTest(
                    test_name="Resource Monitoring Integration",
                    category="integration",
                    description="Validate resource monitoring integration",
                    passed=passed,
                    score=score,
                    execution_time_ms=(time.time() - test_start) * 1000,
                    details=details,
                    warnings=[],
                    errors=[]
                ))
            
        except Exception as e:
            tests.append(ValidationTest(
                test_name="Resource Monitoring Integration",
                category="integration",
                description="Validate resource monitoring integration",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={},
                warnings=[],
                errors=[str(e)]
            ))
        
        return tests
    
    def run_stress_test(self, duration_minutes: float = 5.0) -> ValidationTest:
        """
        Run stress test to validate optimization under sustained load.
        
        Args:
            duration_minutes: Stress test duration in minutes
            
        Returns:
            Stress test validation result
        """
        logger.info(f"Running {duration_minutes}-minute stress test...")
        test_start = time.time()
        
        try:
            # Initialize all components
            if not self.training_pipeline:
                self.training_pipeline = MacBookTrainingPipeline()
                self.training_pipeline.configure_optimization_components()
            
            # Start monitoring
            if self.resource_monitor:
                self.resource_monitor.start_training_monitoring("stress_test")
            
            if self.thermal_monitor:
                self.thermal_monitor.start_monitoring()
            
            if self.email_cpu_optimizer:
                self.email_cpu_optimizer.start_dynamic_scaling()
            
            # Run sustained load
            end_time = time.time() + (duration_minutes * 60)
            iteration = 0
            
            while time.time() < end_time:
                if TORCH_AVAILABLE:
                    # Simulate training workload
                    batch_size = 8
                    seq_len = 512
                    hidden_size = 256
                    
                    # Create tensors to simulate email batch processing
                    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
                    
                    # Simulate forward pass
                    attention_weights = torch.softmax(
                        torch.matmul(hidden_states, hidden_states.transpose(-2, -1)) / (hidden_size ** 0.5),
                        dim=-1
                    )
                    output = torch.matmul(attention_weights, hidden_states)
                    
                    # Simulate classification
                    classifier_weights = torch.randn(hidden_size, 10, requires_grad=True)
                    logits = torch.matmul(output.mean(dim=1), classifier_weights)
                    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 10, (batch_size,)))
                    
                    # Simulate backward pass
                    loss.backward()
                    
                    iteration += 1
                    
                    # Update monitoring
                    if self.resource_monitor:
                        self.resource_monitor.update_training_progress(
                            step=iteration,
                            accuracy=0.85 + (iteration % 100) * 0.001,
                            loss=loss.item(),
                            learning_rate=1e-4,
                            emails_processed=batch_size
                        )
                
                time.sleep(0.1)  # Small delay to prevent overwhelming the system
            
            # Stop monitoring and collect results
            performance_metrics = None
            if self.resource_monitor:
                performance_metrics = self.resource_monitor.stop_training_monitoring()
            
            thermal_summary = None
            if self.thermal_monitor:
                thermal_summary = self.thermal_monitor.get_thermal_summary()
                self.thermal_monitor.stop_monitoring()
            
            cpu_summary = None
            if self.email_cpu_optimizer:
                cpu_summary = self.email_cpu_optimizer.get_cpu_optimization_summary()
                self.email_cpu_optimizer.stop_dynamic_scaling()
            
            # Evaluate stress test results
            passed = True
            score = 100
            warnings = []
            errors = []
            
            # Check for thermal issues
            if thermal_summary and thermal_summary.get("thermal_events", {}).get("throttling_events", 0) > 10:
                score -= 20
                warnings.append("Frequent thermal throttling during stress test")
            
            # Check for memory issues
            if performance_metrics and performance_metrics.memory_pressure_events > 20:
                score -= 15
                warnings.append("Frequent memory pressure events during stress test")
            
            # Check for performance degradation
            if performance_metrics and performance_metrics.average_emails_per_second < 1.0:
                score -= 25
                warnings.append("Low email processing speed during stress test")
                if performance_metrics.average_emails_per_second < 0.5:
                    passed = False
                    errors.append("Critically low performance during stress test")
            
            # Check for system stability
            if performance_metrics and performance_metrics.resource_stability < 0.7:
                score -= 10
                warnings.append("Unstable resource usage during stress test")
            
            details = {
                "duration_minutes": duration_minutes,
                "iterations_completed": iteration,
                "performance_metrics": asdict(performance_metrics) if performance_metrics else None,
                "thermal_summary": thermal_summary,
                "cpu_summary": cpu_summary,
                "final_score": max(0, score)
            }
            
            return ValidationTest(
                test_name="Sustained Load Stress Test",
                category="integration",
                description=f"Validate system stability under {duration_minutes}-minute sustained load",
                passed=passed,
                score=max(0, score),
                execution_time_ms=(time.time() - test_start) * 1000,
                details=details,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return ValidationTest(
                test_name="Sustained Load Stress Test",
                category="integration",
                description=f"Validate system stability under {duration_minutes}-minute sustained load",
                passed=False,
                score=0,
                execution_time_ms=(time.time() - test_start) * 1000,
                details={"error": str(e)},
                warnings=[],
                errors=[str(e)]
            )
    
    def generate_validation_report(self, validation_suite: ValidationSuite) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_suite: Validation suite results
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 80)
        report.append("MACBOOK HARDWARE OPTIMIZATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Grade: {validation_suite.overall_grade}")
        report.append(f"Total Score: {validation_suite.total_score:.1f}/100")
        report.append(f"System Ready for Training: {'YES' if validation_suite.system_ready else 'NO'}")
        report.append(f"Tests Passed: {validation_suite.passed_tests}/{len(validation_suite.tests)}")
        report.append(f"Execution Time: {validation_suite.total_execution_time:.1f} seconds")
        report.append("")
        
        # Category Scores
        report.append("CATEGORY SCORES")
        report.append("-" * 40)
        report.append(f"Memory Management: {validation_suite.memory_score:.1f}/100")
        report.append(f"CPU Optimization: {validation_suite.cpu_score:.1f}/100")
        report.append(f"Thermal Management: {validation_suite.thermal_score:.1f}/100")
        report.append(f"System Integration: {validation_suite.integration_score:.1f}/100")
        report.append("")
        
        # Critical Issues
        if validation_suite.critical_issues:
            report.append("CRITICAL ISSUES")
            report.append("-" * 40)
            for issue in validation_suite.critical_issues:
                report.append(f"• {issue}")
            report.append("")
        
        # Recommendations
        if validation_suite.recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for rec in validation_suite.recommendations:
                report.append(f"• {rec}")
            report.append("")
        
        # Detailed Test Results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 40)
        
        for category in ["memory", "cpu", "thermal", "integration"]:
            category_tests = [t for t in validation_suite.tests if t.category == category]
            if category_tests:
                report.append(f"\n{category.upper()} TESTS:")
                for test in category_tests:
                    status = "PASS" if test.passed else "FAIL"
                    report.append(f"  [{status}] {test.test_name} - Score: {test.score:.1f}/100")
                    if test.warnings:
                        for warning in test.warnings:
                            report.append(f"    WARNING: {warning}")
                    if test.errors:
                        for error in test.errors:
                            report.append(f"    ERROR: {error}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"status": "no_validations", "message": "No validations have been run"}
        
        latest_validation = self.validation_results[-1]
        
        return {
            "status": "completed",
            "latest_validation": {
                "overall_grade": latest_validation.overall_grade,
                "total_score": latest_validation.total_score,
                "system_ready": latest_validation.system_ready,
                "passed_tests": latest_validation.passed_tests,
                "total_tests": len(latest_validation.tests),
                "execution_time": latest_validation.total_execution_time
            },
            "category_scores": {
                "memory": latest_validation.memory_score,
                "cpu": latest_validation.cpu_score,
                "thermal": latest_validation.thermal_score,
                "integration": latest_validation.integration_score
            },
            "issues_and_recommendations": {
                "critical_issues": latest_validation.critical_issues,
                "recommendations": latest_validation.recommendations
            },
            "validation_history": [
                {
                    "timestamp": v.start_time,
                    "grade": v.overall_grade,
                    "score": v.total_score,
                    "system_ready": v.system_ready
                }
                for v in self.validation_results
            ]
        }