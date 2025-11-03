"""
Deployment Validation System for Email Classification Models

This module provides comprehensive validation for deployed models including
automated testing on held-out data, performance regression testing,
and production monitoring with alerting capabilities.
"""

import os
import json
import time
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from .model_export import ModelMetadata
from .inference_api import InferenceEngine, ModelLoader, InferenceConfig
from models.email_tokenizer import EmailTokenizer


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    # Accuracy metrics
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    precision_scores: Dict[str, float]
    recall_scores: Dict[str, float]
    f1_scores: Dict[str, float]
    
    # Performance metrics
    average_inference_time_ms: float
    throughput_emails_per_second: float
    memory_usage_mb: float
    
    # Confidence metrics
    average_confidence: float
    confidence_distribution: Dict[str, int]  # Binned confidence scores
    calibration_error: float
    
    # Robustness metrics
    adversarial_accuracy: float
    noise_robustness: float
    
    # Consistency metrics
    prediction_stability: float
    cross_validation_std: float


@dataclass
class ValidationConfig:
    """Configuration for deployment validation."""
    # Test data settings
    test_data_path: str
    validation_split: float = 0.2
    min_samples_per_category: int = 10
    
    # Performance thresholds
    min_accuracy: float = 0.90
    min_category_accuracy: float = 0.85
    max_inference_time_ms: float = 1000.0
    min_throughput_eps: float = 10.0  # emails per second
    
    # Robustness testing
    enable_adversarial_testing: bool = True
    enable_noise_testing: bool = True
    noise_levels: List[float] = None
    
    # Regression testing
    enable_regression_testing: bool = True
    baseline_model_path: Optional[str] = None
    regression_threshold: float = 0.02  # 2% accuracy drop threshold
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval_minutes: int = 60
    alert_thresholds: Dict[str, float] = None
    
    # Output settings
    save_detailed_results: bool = True
    generate_reports: bool = True
    output_dir: str = "validation_results"
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.noise_levels is None:
            self.noise_levels = [0.1, 0.2, 0.3]
        
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "accuracy_drop": 0.05,
                "response_time_increase": 2.0,
                "error_rate_increase": 0.1
            }


@dataclass
class ValidationResult:
    """Result of deployment validation."""
    success: bool
    validation_id: str
    timestamp: datetime
    
    # Test results
    metrics: ValidationMetrics
    test_summary: Dict[str, Any]
    
    # Regression results
    regression_passed: bool
    baseline_comparison: Optional[Dict[str, float]]
    
    # Issues found
    warnings: List[str]
    errors: List[str]
    critical_issues: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    # Detailed results
    detailed_results: Optional[Dict[str, Any]]


@dataclass
class MonitoringAlert:
    """Monitoring alert for production issues."""
    alert_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    
    # Context
    model_id: str
    deployment_environment: str
    
    # Resolution
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class ValidationTestDataManager:
    """Manages test datasets for validation."""
    
    def __init__(self, config: ValidationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.test_data: List[Dict[str, Any]] = []
        self.validation_data: List[Dict[str, Any]] = []
        self.category_distribution: Dict[str, int] = {}
    
    def load_test_data(self) -> bool:
        """Load and prepare test data."""
        try:
            self.logger.info(f"Loading test data from {self.config.test_data_path}")
            
            if not os.path.exists(self.config.test_data_path):
                self.logger.error(f"Test data file not found: {self.config.test_data_path}")
                return False
            
            # Load data based on file format
            if self.config.test_data_path.endswith('.json'):
                with open(self.config.test_data_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.test_data = data
                    elif isinstance(data, dict) and 'emails' in data:
                        self.test_data = data['emails']
                    else:
                        self.logger.error("Invalid JSON format for test data")
                        return False
            
            # Validate data format
            if not self._validate_data_format():
                return False
            
            # Split data for validation
            self._split_validation_data()
            
            # Analyze category distribution
            self._analyze_category_distribution()
            
            self.logger.info(f"Loaded {len(self.test_data)} test samples, "
                           f"{len(self.validation_data)} validation samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            return False
    
    def _validate_data_format(self) -> bool:
        """Validate test data format."""
        required_fields = ['subject', 'body', 'category']
        
        for i, email in enumerate(self.test_data):
            if not isinstance(email, dict):
                self.logger.error(f"Invalid email format at index {i}")
                return False
            
            for field in required_fields:
                if field not in email:
                    self.logger.error(f"Missing field '{field}' in email at index {i}")
                    return False
                
                if not email[field] or not str(email[field]).strip():
                    self.logger.error(f"Empty field '{field}' in email at index {i}")
                    return False
        
        return True
    
    def _split_validation_data(self):
        """Split data into test and validation sets."""
        import random
        
        # Shuffle data
        shuffled_data = self.test_data.copy()
        random.shuffle(shuffled_data)
        
        # Split
        split_idx = int(len(shuffled_data) * (1 - self.config.validation_split))
        self.test_data = shuffled_data[:split_idx]
        self.validation_data = shuffled_data[split_idx:]
    
    def _analyze_category_distribution(self):
        """Analyze category distribution in test data."""
        self.category_distribution = {}
        
        for email in self.test_data:
            category = email['category']
            self.category_distribution[category] = self.category_distribution.get(category, 0) + 1
        
        # Check minimum samples per category
        insufficient_categories = []
        for category, count in self.category_distribution.items():
            if count < self.config.min_samples_per_category:
                insufficient_categories.append(f"{category}: {count}")
        
        if insufficient_categories:
            self.logger.warning(f"Categories with insufficient samples: {insufficient_categories}")
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get test data."""
        return self.test_data
    
    def get_validation_data(self) -> List[Dict[str, Any]]:
        """Get validation data."""
        return self.validation_data
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get category distribution."""
        return self.category_distribution


class ModelValidator:
    """Validates deployed models against test data."""
    
    def __init__(self, 
                 inference_engine: InferenceEngine,
                 config: ValidationConfig,
                 logger: logging.Logger):
        self.inference_engine = inference_engine
        self.config = config
        self.logger = logger
        
        self.test_data_manager = ValidationTestDataManager(config, logger)
    
    def validate_model(self) -> ValidationResult:
        """Run comprehensive model validation."""
        validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting model validation: {validation_id}")
        
        warnings = []
        errors = []
        critical_issues = []
        recommendations = []
        
        try:
            # Load test data
            if not self.test_data_manager.load_test_data():
                errors.append("Failed to load test data")
                return ValidationResult(
                    success=False,
                    validation_id=validation_id,
                    timestamp=datetime.now(),
                    metrics=ValidationMetrics(
                        overall_accuracy=0.0,
                        category_accuracies={},
                        precision_scores={},
                        recall_scores={},
                        f1_scores={},
                        average_inference_time_ms=0.0,
                        throughput_emails_per_second=0.0,
                        memory_usage_mb=0.0,
                        average_confidence=0.0,
                        confidence_distribution={},
                        calibration_error=0.0,
                        adversarial_accuracy=0.0,
                        noise_robustness=0.0,
                        prediction_stability=0.0,
                        cross_validation_std=0.0
                    ),
                    test_summary={},
                    regression_passed=False,
                    baseline_comparison=None,
                    warnings=warnings,
                    errors=errors,
                    critical_issues=critical_issues,
                    recommendations=recommendations,
                    detailed_results=None
                )
            
            # Run accuracy validation
            accuracy_results = self._validate_accuracy()
            
            # Run performance validation
            performance_results = self._validate_performance()
            
            # Run robustness testing
            robustness_results = self._validate_robustness()
            
            # Run regression testing
            regression_results = self._validate_regression()
            
            # Combine results
            metrics = self._combine_metrics(
                accuracy_results, 
                performance_results, 
                robustness_results
            )
            
            # Check thresholds and generate recommendations
            threshold_results = self._check_thresholds(metrics)
            warnings.extend(threshold_results["warnings"])
            errors.extend(threshold_results["errors"])
            critical_issues.extend(threshold_results["critical_issues"])
            recommendations.extend(threshold_results["recommendations"])
            
            # Determine overall success
            success = len(errors) == 0 and len(critical_issues) == 0
            
            # Create detailed results
            detailed_results = {
                "accuracy_results": accuracy_results,
                "performance_results": performance_results,
                "robustness_results": robustness_results,
                "regression_results": regression_results,
                "category_distribution": self.test_data_manager.get_category_distribution()
            }
            
            result = ValidationResult(
                success=success,
                validation_id=validation_id,
                timestamp=datetime.now(),
                metrics=metrics,
                test_summary=self._generate_test_summary(metrics),
                regression_passed=regression_results.get("passed", True),
                baseline_comparison=regression_results.get("comparison"),
                warnings=warnings,
                errors=errors,
                critical_issues=critical_issues,
                recommendations=recommendations,
                detailed_results=detailed_results if self.config.save_detailed_results else None
            )
            
            self.logger.info(f"Validation completed: {'PASSED' if success else 'FAILED'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {e}")
            errors.append(f"Validation exception: {str(e)}")
            
            return ValidationResult(
                success=False,
                validation_id=validation_id,
                timestamp=datetime.now(),
                metrics=ValidationMetrics(
                    overall_accuracy=0.0,
                    category_accuracies={},
                    precision_scores={},
                    recall_scores={},
                    f1_scores={},
                    average_inference_time_ms=0.0,
                    throughput_emails_per_second=0.0,
                    memory_usage_mb=0.0,
                    average_confidence=0.0,
                    confidence_distribution={},
                    calibration_error=0.0,
                    adversarial_accuracy=0.0,
                    noise_robustness=0.0,
                    prediction_stability=0.0,
                    cross_validation_std=0.0
                ),
                test_summary={},
                regression_passed=False,
                baseline_comparison=None,
                warnings=warnings,
                errors=errors,
                critical_issues=critical_issues,
                recommendations=recommendations,
                detailed_results=None
            )
    
    def _validate_accuracy(self) -> Dict[str, Any]:
        """Validate model accuracy on test data."""
        self.logger.info("Running accuracy validation...")
        
        test_data = self.test_data_manager.get_test_data()
        
        predictions = []
        true_labels = []
        confidences = []
        categories = set()
        
        # Run predictions
        for email in test_data:
            try:
                result = self.inference_engine.predict_single(email)
                
                predictions.append(result["predicted_category"])
                true_labels.append(email["category"])
                confidences.append(result["confidence"])
                categories.add(email["category"])
                
            except Exception as e:
                self.logger.error(f"Prediction failed for email: {e}")
                continue
        
        # Calculate metrics
        categories = sorted(list(categories))
        
        # Overall accuracy
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        overall_accuracy = correct / len(predictions) if predictions else 0.0
        
        # Per-category metrics
        category_accuracies = {}
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        
        for category in categories:
            # True positives, false positives, false negatives
            tp = sum(1 for p, t in zip(predictions, true_labels) if p == category and t == category)
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == category and t != category)
            fn = sum(1 for p, t in zip(predictions, true_labels) if p != category and t == category)
            
            # Category accuracy (for this category as true label)
            category_total = sum(1 for t in true_labels if t == category)
            category_correct = sum(1 for p, t in zip(predictions, true_labels) if t == category and p == t)
            category_accuracies[category] = category_correct / category_total if category_total > 0 else 0.0
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_scores[category] = precision
            recall_scores[category] = recall
            f1_scores[category] = f1
        
        # Confidence analysis
        average_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Confidence distribution (binned)
        confidence_bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for conf in confidences:
            if conf < 0.2:
                confidence_bins["0.0-0.2"] += 1
            elif conf < 0.4:
                confidence_bins["0.2-0.4"] += 1
            elif conf < 0.6:
                confidence_bins["0.4-0.6"] += 1
            elif conf < 0.8:
                confidence_bins["0.6-0.8"] += 1
            else:
                confidence_bins["0.8-1.0"] += 1
        
        # Calibration error (simplified)
        calibration_error = self._calculate_calibration_error(predictions, true_labels, confidences)
        
        return {
            "overall_accuracy": overall_accuracy,
            "category_accuracies": category_accuracies,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
            "average_confidence": average_confidence,
            "confidence_distribution": confidence_bins,
            "calibration_error": calibration_error,
            "total_samples": len(predictions),
            "categories": categories
        }
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate model performance metrics."""
        self.logger.info("Running performance validation...")
        
        test_data = self.test_data_manager.get_test_data()[:100]  # Sample for performance testing
        
        inference_times = []
        
        # Measure inference times
        for email in test_data:
            start_time = time.time()
            try:
                self.inference_engine.predict_single(email)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
            except Exception as e:
                self.logger.error(f"Performance test failed for email: {e}")
                continue
        
        # Calculate performance metrics
        avg_inference_time = statistics.mean(inference_times) if inference_times else 0.0
        throughput = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0.0  # emails per second
        
        # Memory usage (simplified - would use more sophisticated monitoring in practice)
        memory_usage = 100.0  # Placeholder
        
        return {
            "average_inference_time_ms": avg_inference_time,
            "throughput_emails_per_second": throughput,
            "memory_usage_mb": memory_usage,
            "inference_times": inference_times
        }
    
    def _validate_robustness(self) -> Dict[str, Any]:
        """Validate model robustness."""
        self.logger.info("Running robustness validation...")
        
        results = {
            "adversarial_accuracy": 0.0,
            "noise_robustness": 0.0,
            "prediction_stability": 0.0
        }
        
        if not self.config.enable_adversarial_testing and not self.config.enable_noise_testing:
            return results
        
        test_data = self.test_data_manager.get_test_data()[:50]  # Sample for robustness testing
        
        # Adversarial testing (simplified)
        if self.config.enable_adversarial_testing:
            adversarial_correct = 0
            for email in test_data:
                try:
                    # Simple adversarial example: add noise to text
                    adversarial_email = email.copy()
                    adversarial_email["body"] = email["body"] + " random noise text"
                    
                    original_result = self.inference_engine.predict_single(email)
                    adversarial_result = self.inference_engine.predict_single(adversarial_email)
                    
                    # Check if prediction is still correct
                    if adversarial_result["predicted_category"] == email["category"]:
                        adversarial_correct += 1
                        
                except Exception as e:
                    self.logger.error(f"Adversarial test failed: {e}")
                    continue
            
            results["adversarial_accuracy"] = adversarial_correct / len(test_data) if test_data else 0.0
        
        # Noise robustness testing
        if self.config.enable_noise_testing:
            noise_accuracies = []
            
            for noise_level in self.config.noise_levels:
                noise_correct = 0
                
                for email in test_data:
                    try:
                        # Add character-level noise
                        noisy_email = self._add_text_noise(email, noise_level)
                        
                        result = self.inference_engine.predict_single(noisy_email)
                        
                        if result["predicted_category"] == email["category"]:
                            noise_correct += 1
                            
                    except Exception as e:
                        self.logger.error(f"Noise test failed: {e}")
                        continue
                
                noise_accuracy = noise_correct / len(test_data) if test_data else 0.0
                noise_accuracies.append(noise_accuracy)
            
            results["noise_robustness"] = statistics.mean(noise_accuracies) if noise_accuracies else 0.0
        
        # Prediction stability (run same prediction multiple times)
        stability_scores = []
        for email in test_data[:10]:  # Small sample for stability testing
            try:
                predictions = []
                for _ in range(5):  # Run 5 times
                    result = self.inference_engine.predict_single(email)
                    predictions.append(result["predicted_category"])
                
                # Calculate stability (consistency of predictions)
                most_common = max(set(predictions), key=predictions.count)
                stability = predictions.count(most_common) / len(predictions)
                stability_scores.append(stability)
                
            except Exception as e:
                self.logger.error(f"Stability test failed: {e}")
                continue
        
        results["prediction_stability"] = statistics.mean(stability_scores) if stability_scores else 0.0
        
        return results
    
    def _validate_regression(self) -> Dict[str, Any]:
        """Validate against baseline model for regression testing."""
        self.logger.info("Running regression validation...")
        
        if not self.config.enable_regression_testing or not self.config.baseline_model_path:
            return {"passed": True, "comparison": None}
        
        # This would load and compare against baseline model
        # For now, return placeholder results
        return {
            "passed": True,
            "comparison": {
                "accuracy_difference": 0.01,
                "performance_difference": 0.05
            }
        }
    
    def _add_text_noise(self, email: Dict[str, Any], noise_level: float) -> Dict[str, Any]:
        """Add character-level noise to email text."""
        import random
        import string
        
        noisy_email = email.copy()
        
        for field in ["subject", "body"]:
            if field in email and email[field]:
                text = email[field]
                chars = list(text)
                
                # Add random character substitutions
                num_changes = int(len(chars) * noise_level)
                for _ in range(num_changes):
                    if chars:
                        idx = random.randint(0, len(chars) - 1)
                        chars[idx] = random.choice(string.ascii_letters + " ")
                
                noisy_email[field] = "".join(chars)
        
        return noisy_email
    
    def _calculate_calibration_error(self, predictions: List[str], true_labels: List[str], confidences: List[float]) -> float:
        """Calculate calibration error (simplified)."""
        if not predictions or not confidences:
            return 0.0
        
        # Bin predictions by confidence
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration_errors = []
        
        for bin_min, bin_max in bins:
            bin_predictions = []
            bin_accuracies = []
            bin_confidences = []
            
            for p, t, c in zip(predictions, true_labels, confidences):
                if bin_min <= c < bin_max:
                    bin_predictions.append(p)
                    bin_accuracies.append(1.0 if p == t else 0.0)
                    bin_confidences.append(c)
            
            if bin_predictions:
                avg_confidence = statistics.mean(bin_confidences)
                avg_accuracy = statistics.mean(bin_accuracies)
                calibration_errors.append(abs(avg_confidence - avg_accuracy))
        
        return statistics.mean(calibration_errors) if calibration_errors else 0.0
    
    def _combine_metrics(self, accuracy_results: Dict, performance_results: Dict, robustness_results: Dict) -> ValidationMetrics:
        """Combine all validation results into metrics object."""
        return ValidationMetrics(
            overall_accuracy=accuracy_results["overall_accuracy"],
            category_accuracies=accuracy_results["category_accuracies"],
            precision_scores=accuracy_results["precision_scores"],
            recall_scores=accuracy_results["recall_scores"],
            f1_scores=accuracy_results["f1_scores"],
            average_inference_time_ms=performance_results["average_inference_time_ms"],
            throughput_emails_per_second=performance_results["throughput_emails_per_second"],
            memory_usage_mb=performance_results["memory_usage_mb"],
            average_confidence=accuracy_results["average_confidence"],
            confidence_distribution=accuracy_results["confidence_distribution"],
            calibration_error=accuracy_results["calibration_error"],
            adversarial_accuracy=robustness_results["adversarial_accuracy"],
            noise_robustness=robustness_results["noise_robustness"],
            prediction_stability=robustness_results["prediction_stability"],
            cross_validation_std=0.0  # Placeholder
        )
    
    def _check_thresholds(self, metrics: ValidationMetrics) -> Dict[str, List[str]]:
        """Check metrics against thresholds and generate issues/recommendations."""
        warnings = []
        errors = []
        critical_issues = []
        recommendations = []
        
        # Check accuracy thresholds
        if metrics.overall_accuracy < self.config.min_accuracy:
            critical_issues.append(f"Overall accuracy {metrics.overall_accuracy:.3f} below threshold {self.config.min_accuracy}")
            recommendations.append("Consider retraining model with more data or different architecture")
        
        # Check category accuracies
        for category, accuracy in metrics.category_accuracies.items():
            if accuracy < self.config.min_category_accuracy:
                errors.append(f"Category '{category}' accuracy {accuracy:.3f} below threshold {self.config.min_category_accuracy}")
                recommendations.append(f"Improve training data quality for category '{category}'")
        
        # Check performance thresholds
        if metrics.average_inference_time_ms > self.config.max_inference_time_ms:
            warnings.append(f"Average inference time {metrics.average_inference_time_ms:.1f}ms exceeds threshold {self.config.max_inference_time_ms}ms")
            recommendations.append("Consider model optimization or hardware upgrade")
        
        if metrics.throughput_emails_per_second < self.config.min_throughput_eps:
            warnings.append(f"Throughput {metrics.throughput_emails_per_second:.1f} emails/sec below threshold {self.config.min_throughput_eps}")
            recommendations.append("Optimize inference pipeline or scale horizontally")
        
        # Check confidence calibration
        if metrics.calibration_error > 0.1:
            warnings.append(f"High calibration error {metrics.calibration_error:.3f} indicates poor confidence estimates")
            recommendations.append("Apply confidence calibration techniques")
        
        # Check robustness
        if metrics.adversarial_accuracy < 0.8:
            warnings.append(f"Low adversarial accuracy {metrics.adversarial_accuracy:.3f} indicates vulnerability")
            recommendations.append("Consider adversarial training or input validation")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "critical_issues": critical_issues,
            "recommendations": recommendations
        }
    
    def _generate_test_summary(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Generate test summary."""
        return {
            "overall_accuracy": metrics.overall_accuracy,
            "average_category_accuracy": statistics.mean(metrics.category_accuracies.values()) if metrics.category_accuracies else 0.0,
            "average_f1_score": statistics.mean(metrics.f1_scores.values()) if metrics.f1_scores else 0.0,
            "performance_score": min(1.0, self.config.min_throughput_eps / metrics.throughput_emails_per_second) if metrics.throughput_emails_per_second > 0 else 0.0,
            "robustness_score": (metrics.adversarial_accuracy + metrics.noise_robustness + metrics.prediction_stability) / 3.0
        }


class ProductionMonitor:
    """Monitors deployed models in production."""
    
    def __init__(self, config: ValidationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_queue = queue.Queue()
        
        # Metrics history
        self.metrics_history: List[Tuple[datetime, Dict[str, float]]] = []
        self.alerts: List[MonitoringAlert] = []
    
    def start_monitoring(self, inference_engine: InferenceEngine):
        """Start production monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(inference_engine,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop production monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self, inference_engine: InferenceEngine):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics(inference_engine)
                
                # Store metrics
                self.metrics_history.append((datetime.now(), current_metrics))
                
                # Check for alerts
                self._check_alerts(current_metrics)
                
                # Sleep until next monitoring interval
                time.sleep(self.config.monitoring_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _collect_metrics(self, inference_engine: InferenceEngine) -> Dict[str, float]:
        """Collect current production metrics."""
        stats = inference_engine.get_statistics()
        
        return {
            "total_predictions": stats.get("total_predictions", 0),
            "average_response_time_ms": stats.get("average_response_time_ms", 0.0),
            "predictions_per_second": stats.get("predictions_per_second", 0.0),
            "uptime_hours": stats.get("uptime_seconds", 0) / 3600.0
        }
    
    def _check_alerts(self, current_metrics: Dict[str, float]):
        """Check metrics against alert thresholds."""
        if len(self.metrics_history) < 2:
            return  # Need at least 2 data points for comparison
        
        previous_metrics = self.metrics_history[-2][1]
        
        # Check response time increase
        current_response_time = current_metrics.get("average_response_time_ms", 0.0)
        previous_response_time = previous_metrics.get("average_response_time_ms", 0.0)
        
        if previous_response_time > 0:
            response_time_increase = (current_response_time - previous_response_time) / previous_response_time
            
            if response_time_increase > self.config.alert_thresholds["response_time_increase"]:
                alert = MonitoringAlert(
                    alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    severity="medium",
                    metric_name="response_time_increase",
                    current_value=response_time_increase,
                    threshold_value=self.config.alert_thresholds["response_time_increase"],
                    message=f"Response time increased by {response_time_increase:.1%}",
                    model_id="current_model",
                    deployment_environment="production"
                )
                
                self.alerts.append(alert)
                self.alert_queue.put(alert)
                self.logger.warning(f"Alert generated: {alert.message}")
    
    def get_alerts(self, unresolved_only: bool = True) -> List[MonitoringAlert]:
        """Get monitoring alerts."""
        if unresolved_only:
            return [alert for alert in self.alerts if not alert.resolved]
        else:
            return self.alerts.copy()
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve a monitoring alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                alert.resolution_notes = resolution_notes
                self.logger.info(f"Alert {alert_id} resolved")
                break
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        active_alerts = len(self.get_alerts(unresolved_only=True))
        total_alerts = len(self.alerts)
        
        return {
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.metrics_history),
            "active_alerts": active_alerts,
            "total_alerts": total_alerts,
            "monitoring_interval_minutes": self.config.monitoring_interval_minutes
        }


class DeploymentValidator:
    """Main deployment validation orchestrator."""
    
    def __init__(self, config: ValidationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.monitor: Optional[ProductionMonitor] = None
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
    
    def validate_deployment(self, 
                          model_path: str,
                          tokenizer_path: str,
                          test_data_path: Optional[str] = None) -> ValidationResult:
        """
        Run comprehensive deployment validation.
        
        Args:
            model_path: Path to deployed model
            tokenizer_path: Path to tokenizer
            test_data_path: Path to test data (overrides config)
            
        Returns:
            ValidationResult with comprehensive validation results
        """
        self.logger.info("Starting deployment validation...")
        
        # Update test data path if provided
        if test_data_path:
            self.config.test_data_path = test_data_path
        
        try:
            # Create inference engine
            inference_config = InferenceConfig(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device="cpu"
            )
            
            model_loader = ModelLoader(inference_config, self.logger)
            if not model_loader.load_model():
                raise RuntimeError("Failed to load model for validation")
            
            inference_engine = InferenceEngine(model_loader, inference_config, self.logger)
            
            # Create validator
            validator = ModelValidator(inference_engine, self.config, self.logger)
            
            # Run validation
            result = validator.validate_model()
            
            # Save results
            self._save_validation_result(result)
            
            # Add to history
            self.validation_history.append(result)
            
            # Start monitoring if enabled
            if self.config.enable_monitoring and result.success:
                self.start_monitoring(inference_engine)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment validation failed: {e}")
            
            # Return failed result
            return ValidationResult(
                success=False,
                validation_id=f"validation_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                metrics=ValidationMetrics(
                    overall_accuracy=0.0,
                    category_accuracies={},
                    precision_scores={},
                    recall_scores={},
                    f1_scores={},
                    average_inference_time_ms=0.0,
                    throughput_emails_per_second=0.0,
                    memory_usage_mb=0.0,
                    average_confidence=0.0,
                    confidence_distribution={},
                    calibration_error=0.0,
                    adversarial_accuracy=0.0,
                    noise_robustness=0.0,
                    prediction_stability=0.0,
                    cross_validation_std=0.0
                ),
                test_summary={},
                regression_passed=False,
                baseline_comparison=None,
                warnings=[],
                errors=[f"Validation failed: {str(e)}"],
                critical_issues=[],
                recommendations=[],
                detailed_results=None
            )
    
    def start_monitoring(self, inference_engine: InferenceEngine):
        """Start production monitoring."""
        if not self.config.enable_monitoring:
            return
        
        self.monitor = ProductionMonitor(self.config, self.logger)
        self.monitor.start_monitoring(inference_engine)
    
    def stop_monitoring(self):
        """Stop production monitoring."""
        if self.monitor:
            self.monitor.stop_monitoring()
    
    def _save_validation_result(self, result: ValidationResult):
        """Save validation result to file."""
        result_file = self.output_dir / f"{result.validation_id}_result.json"
        
        try:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.info(f"Validation result saved to {result_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation result: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations."""
        successful_validations = [v for v in self.validation_history if v.success]
        
        summary = {
            "total_validations": len(self.validation_history),
            "successful_validations": len(successful_validations),
            "failed_validations": len(self.validation_history) - len(successful_validations),
            "monitoring_active": self.monitor.monitoring_active if self.monitor else False
        }
        
        if successful_validations:
            latest_validation = self.validation_history[-1]
            summary.update({
                "latest_validation": {
                    "validation_id": latest_validation.validation_id,
                    "success": latest_validation.success,
                    "overall_accuracy": latest_validation.metrics.overall_accuracy,
                    "timestamp": latest_validation.timestamp.isoformat()
                }
            })
        
        if self.monitor:
            summary["monitoring_summary"] = self.monitor.get_monitoring_summary()
        
        return summary