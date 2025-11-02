"""
Configuration validation module for MacBook TRM training optimization.

This module provides comprehensive validation of training configurations
against MacBook hardware constraints, with automatic correction suggestions
and warnings for invalid configurations.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import logging

from .hardware_detection import HardwareDetector, CPUSpecs, MemorySpecs, PlatformCapabilities
from .training_config_adapter import HardwareSpecs, TrainingParams

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    level: ValidationLevel
    category: str
    message: str
    current_value: Any
    suggested_value: Optional[Any] = None
    auto_correctable: bool = False
    impact: str = ""


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: List[ValidationIssue]
    corrected_config: Optional[Dict[str, Any]] = None
    performance_impact: str = ""
    compatibility_score: float = 0.0  # 0-100 scale


@dataclass
class ConfigurationLimits:
    """Hardware-based configuration limits."""
    max_batch_size: int
    min_batch_size: int
    max_memory_mb: int
    max_workers: int
    max_torch_threads: int
    recommended_sequence_length: int
    max_gradient_accumulation: int


class ConfigurationValidator:
    """Validator for MacBook TRM training configurations."""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        """
        Initialize configuration validator.
        
        Args:
            hardware_detector: Hardware detector instance (created if None)
        """
        self.hardware_detector = hardware_detector or HardwareDetector()
        self._hardware_specs = None
        self._limits = None
        
    def get_hardware_specs(self) -> HardwareSpecs:
        """Get cached hardware specifications."""
        if self._hardware_specs is None:
            cpu_specs = self.hardware_detector.detect_cpu_specs()
            memory_specs = self.hardware_detector.detect_memory_specs()
            platform_caps = self.hardware_detector.detect_platform_capabilities()
            optimal_workers = self.hardware_detector.get_optimal_worker_count()
            hardware_summary = self.hardware_detector.get_hardware_summary()
            
            self._hardware_specs = HardwareSpecs(
                cpu=cpu_specs,
                memory=memory_specs,
                platform=platform_caps,
                optimal_workers=optimal_workers,
                hardware_summary=hardware_summary
            )
            
        return self._hardware_specs
    
    def calculate_hardware_limits(self, hardware_specs: Optional[HardwareSpecs] = None) -> ConfigurationLimits:
        """
        Calculate hardware-based configuration limits.
        
        Args:
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Configuration limits based on hardware
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
            
        if self._limits is not None:
            return self._limits
        
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        memory_mb = hardware_specs.memory.available_memory / (1024**2)
        
        # Calculate memory limits
        # Reserve 2GB for system and other processes
        usable_memory_mb = max(1000, memory_mb - 2048)
        max_memory_mb = int(usable_memory_mb * 0.9)  # Use 90% of usable memory
        
        # Estimate memory per sample for 7M parameter model
        model_base_memory_mb = 200  # Base model memory
        memory_per_sample_mb = 10   # Conservative estimate per sample
        
        # Calculate batch size limits
        max_batch_size = max(1, int((max_memory_mb - model_base_memory_mb) / memory_per_sample_mb))
        max_batch_size = min(max_batch_size, 64)  # Cap at 64 for stability
        min_batch_size = 1
        
        # CPU limits
        max_workers = min(hardware_specs.cpu.cores, 8)  # Cap at 8 workers
        max_torch_threads = min(hardware_specs.cpu.threads, 16)  # Cap at 16 threads
        
        # Sequence length based on memory
        if memory_gb < 6:
            recommended_sequence_length = 256
        elif memory_gb < 12:
            recommended_sequence_length = 512
        else:
            recommended_sequence_length = 1024
        
        # Gradient accumulation limits
        max_gradient_accumulation = min(32, 512 // min_batch_size)  # Target effective batch of 512
        
        self._limits = ConfigurationLimits(
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            max_memory_mb=max_memory_mb,
            max_workers=max_workers,
            max_torch_threads=max_torch_threads,
            recommended_sequence_length=recommended_sequence_length,
            max_gradient_accumulation=max_gradient_accumulation
        )
        
        return self._limits
    
    def validate_configuration(self, config: Dict[str, Any], 
                             auto_correct: bool = False) -> ValidationResult:
        """
        Validate training configuration against hardware constraints.
        
        Args:
            config: Training configuration to validate
            auto_correct: Whether to automatically correct issues
            
        Returns:
            Validation result with issues and corrections
        """
        hardware_specs = self.get_hardware_specs()
        limits = self.calculate_hardware_limits(hardware_specs)
        issues = []
        corrected_config = config.copy() if auto_correct else None
        
        # Validate batch size
        batch_size_issues = self._validate_batch_size(config, limits)
        issues.extend(batch_size_issues)
        
        # Validate memory settings
        memory_issues = self._validate_memory_settings(config, limits, hardware_specs)
        issues.extend(memory_issues)
        
        # Validate CPU settings
        cpu_issues = self._validate_cpu_settings(config, limits, hardware_specs)
        issues.extend(cpu_issues)
        
        # Validate training parameters
        training_issues = self._validate_training_parameters(config, limits)
        issues.extend(training_issues)
        
        # Validate model parameters
        model_issues = self._validate_model_parameters(config, limits, hardware_specs)
        issues.extend(model_issues)
        
        # Validate optimization settings
        optimization_issues = self._validate_optimization_settings(config, hardware_specs)
        issues.extend(optimization_issues)
        
        # Apply auto-corrections if requested
        if auto_correct and corrected_config is not None:
            corrected_config = self._apply_corrections(corrected_config, issues, limits)
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility_score(issues)
        
        # Determine overall validity
        critical_errors = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        is_valid = len(critical_errors) == 0 and len(errors) == 0
        
        # Generate performance impact assessment
        performance_impact = self._assess_performance_impact(issues, hardware_specs)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            corrected_config=corrected_config,
            performance_impact=performance_impact,
            compatibility_score=compatibility_score
        )
    
    def _validate_batch_size(self, config: Dict[str, Any], 
                           limits: ConfigurationLimits) -> List[ValidationIssue]:
        """Validate batch size configuration."""
        issues = []
        batch_size = config.get('global_batch_size', config.get('batch_size', 8))
        
        if batch_size > limits.max_batch_size:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="memory",
                message=f"Batch size {batch_size} exceeds memory limit",
                current_value=batch_size,
                suggested_value=limits.max_batch_size,
                auto_correctable=True,
                impact="May cause out-of-memory errors"
            ))
        elif batch_size < limits.min_batch_size:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="training",
                message=f"Batch size {batch_size} is too small",
                current_value=batch_size,
                suggested_value=limits.min_batch_size,
                auto_correctable=True,
                impact="Invalid configuration"
            ))
        elif batch_size < 4:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="performance",
                message=f"Small batch size {batch_size} may slow convergence",
                current_value=batch_size,
                suggested_value=min(8, limits.max_batch_size),
                auto_correctable=False,
                impact="Slower training convergence"
            ))
        
        return issues
    
    def _validate_memory_settings(self, config: Dict[str, Any], 
                                limits: ConfigurationLimits,
                                hardware_specs: HardwareSpecs) -> List[ValidationIssue]:
        """Validate memory-related settings."""
        issues = []
        
        # Check memory limit setting
        memory_limit = config.get('memory_limit_mb', limits.max_memory_mb)
        available_mb = hardware_specs.memory.available_memory / (1024**2)
        
        if memory_limit > available_mb * 0.95:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="memory",
                message=f"Memory limit {memory_limit}MB exceeds available memory",
                current_value=memory_limit,
                suggested_value=int(available_mb * 0.8),
                auto_correctable=True,
                impact="System instability and crashes"
            ))
        
        # Check pin_memory setting
        pin_memory = config.get('pin_memory', False)
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        if pin_memory and memory_gb < 12:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="memory",
                message="pin_memory=True not recommended for systems with <12GB RAM",
                current_value=pin_memory,
                suggested_value=False,
                auto_correctable=True,
                impact="Increased memory pressure"
            ))
        
        return issues
    
    def _validate_cpu_settings(self, config: Dict[str, Any], 
                             limits: ConfigurationLimits,
                             hardware_specs: HardwareSpecs) -> List[ValidationIssue]:
        """Validate CPU-related settings."""
        issues = []
        
        # Validate number of workers
        num_workers = config.get('num_workers', 1)
        if num_workers > limits.max_workers:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="cpu",
                message=f"Number of workers {num_workers} exceeds CPU cores",
                current_value=num_workers,
                suggested_value=limits.max_workers,
                auto_correctable=True,
                impact="CPU oversubscription and context switching overhead"
            ))
        
        # Validate torch threads
        torch_threads = config.get('torch_threads', hardware_specs.cpu.cores)
        if torch_threads > limits.max_torch_threads:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="cpu",
                message=f"PyTorch threads {torch_threads} may cause oversubscription",
                current_value=torch_threads,
                suggested_value=min(hardware_specs.cpu.cores, 8),
                auto_correctable=True,
                impact="Reduced performance due to thread contention"
            ))
        
        # Check MKL usage
        use_mkl = config.get('use_mkl', False)
        if use_mkl and not hardware_specs.platform.has_mkl:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="optimization",
                message="MKL optimization requested but not available",
                current_value=use_mkl,
                suggested_value=False,
                auto_correctable=True,
                impact="Configuration error - MKL not available"
            ))
        
        return issues
    
    def _validate_training_parameters(self, config: Dict[str, Any], 
                                    limits: ConfigurationLimits) -> List[ValidationIssue]:
        """Validate training parameters."""
        issues = []
        
        # Validate gradient accumulation
        grad_accum = config.get('gradient_accumulation_steps', 1)
        if grad_accum > limits.max_gradient_accumulation:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="training",
                message=f"High gradient accumulation {grad_accum} may slow training",
                current_value=grad_accum,
                suggested_value=limits.max_gradient_accumulation,
                auto_correctable=False,
                impact="Slower training due to infrequent updates"
            ))
        
        # Validate learning rate
        lr = config.get('lr', 1e-4)
        # Ensure lr is a float (might come as string from YAML)
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                lr = 1e-4  # Default fallback
        
        batch_size = config.get('global_batch_size', config.get('batch_size', 8))
        effective_batch = batch_size * grad_accum
        
        # Check if learning rate is scaled appropriately for batch size
        if effective_batch > 0:  # Avoid math domain error
            expected_lr = 1e-4 * math.sqrt(effective_batch / 64)
            if abs(lr - expected_lr) / expected_lr > 0.5:  # More than 50% difference
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    category="training",
                    message=f"Learning rate may not be optimally scaled for batch size",
                    current_value=lr,
                    suggested_value=expected_lr,
                    auto_correctable=False,
                    impact="Suboptimal convergence"
                ))
        
        return issues
    
    def _validate_model_parameters(self, config: Dict[str, Any], 
                                 limits: ConfigurationLimits,
                                 hardware_specs: HardwareSpecs) -> List[ValidationIssue]:
        """Validate model parameters."""
        issues = []
        
        # Validate sequence length
        seq_len = config.get('seq_len', 512)
        if seq_len > limits.recommended_sequence_length:
            memory_gb = hardware_specs.memory.available_memory / (1024**3)
            if memory_gb < 8:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="memory",
                    message=f"Sequence length {seq_len} may be too large for available memory",
                    current_value=seq_len,
                    suggested_value=limits.recommended_sequence_length,
                    auto_correctable=False,
                    impact="Potential memory issues during training"
                ))
        
        # Validate model size parameters
        model_dim = config.get('model_dim', config.get('d_model', 512))
        num_layers = config.get('num_layers', config.get('n_layers', 6))
        
        # Estimate model parameters
        estimated_params = model_dim * model_dim * num_layers * 4  # Rough estimate
        if estimated_params > 10_000_000:  # More than 10M parameters
            memory_gb = hardware_specs.memory.available_memory / (1024**3)
            if memory_gb < 12:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category="memory",
                    message=f"Large model (~{estimated_params/1e6:.1f}M params) for available memory",
                    current_value=f"{estimated_params/1e6:.1f}M",
                    suggested_value="Consider reducing model_dim or num_layers",
                    auto_correctable=False,
                    impact="High memory usage, potential OOM errors"
                ))
        
        return issues
    
    def _validate_optimization_settings(self, config: Dict[str, Any], 
                                      hardware_specs: HardwareSpecs) -> List[ValidationIssue]:
        """Validate optimization settings."""
        issues = []
        
        # Check device setting
        device = config.get('device', 'cpu')
        if device != 'cpu':
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="hardware",
                message=f"Device '{device}' not supported on MacBook CPU training",
                current_value=device,
                suggested_value='cpu',
                auto_correctable=True,
                impact="Training will fail"
            ))
        
        # Check mixed precision
        mixed_precision = config.get('enable_mixed_precision', False)
        if mixed_precision:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="optimization",
                message="Mixed precision not recommended for CPU training",
                current_value=mixed_precision,
                suggested_value=False,
                auto_correctable=True,
                impact="Potential numerical instability"
            ))
        
        # Check compilation settings
        compile_model = config.get('compile_model', False)
        if compile_model and not hardware_specs.platform.supports_avx2:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                category="optimization",
                message="Model compilation may not provide benefits without AVX2",
                current_value=compile_model,
                suggested_value=False,
                auto_correctable=False,
                impact="Minimal performance impact"
            ))
        
        return issues
    
    def _apply_corrections(self, config: Dict[str, Any], 
                         issues: List[ValidationIssue],
                         limits: ConfigurationLimits) -> Dict[str, Any]:
        """Apply automatic corrections to configuration."""
        corrected_config = config.copy()
        
        for issue in issues:
            if issue.auto_correctable and issue.suggested_value is not None:
                # Apply correction based on issue category
                if issue.category == "memory" and "batch_size" in issue.message.lower():
                    corrected_config['global_batch_size'] = issue.suggested_value
                elif issue.category == "memory" and "memory_limit" in issue.message.lower():
                    corrected_config['memory_limit_mb'] = issue.suggested_value
                elif issue.category == "memory" and "pin_memory" in issue.message.lower():
                    corrected_config['pin_memory'] = issue.suggested_value
                elif issue.category == "cpu" and "workers" in issue.message.lower():
                    corrected_config['num_workers'] = issue.suggested_value
                elif issue.category == "cpu" and "threads" in issue.message.lower():
                    corrected_config['torch_threads'] = issue.suggested_value
                elif issue.category == "optimization" and "mkl" in issue.message.lower():
                    corrected_config['use_mkl'] = issue.suggested_value
                elif issue.category == "hardware" and "device" in issue.message.lower():
                    corrected_config['device'] = issue.suggested_value
                elif issue.category == "optimization" and "mixed_precision" in issue.message.lower():
                    corrected_config['enable_mixed_precision'] = issue.suggested_value
        
        return corrected_config
    
    def _calculate_compatibility_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate compatibility score (0-100) based on validation issues."""
        if not issues:
            return 100.0
        
        # Weight different issue levels
        weights = {
            ValidationLevel.CRITICAL: -30,
            ValidationLevel.ERROR: -20,
            ValidationLevel.WARNING: -10,
            ValidationLevel.INFO: -2
        }
        
        score = 100.0
        for issue in issues:
            score += weights.get(issue.level, 0)
        
        return max(0.0, min(100.0, score))
    
    def _assess_performance_impact(self, issues: List[ValidationIssue], 
                                 hardware_specs: HardwareSpecs) -> str:
        """Assess overall performance impact of configuration issues."""
        if not issues:
            return "Configuration is well-optimized for your hardware."
        
        critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        error_issues = [i for i in issues if i.level == ValidationLevel.ERROR]
        warning_issues = [i for i in issues if i.level == ValidationLevel.WARNING]
        
        if critical_issues:
            return "Critical issues detected - training may fail or cause system instability."
        elif error_issues:
            return "Configuration errors detected - training may not work correctly."
        elif len(warning_issues) > 3:
            return "Multiple performance warnings - training will work but may be suboptimal."
        elif warning_issues:
            return "Minor performance issues detected - consider addressing for better performance."
        else:
            return "Configuration has minor optimization opportunities."
    
    def suggest_configuration_improvements(self, config: Dict[str, Any]) -> List[str]:
        """
        Suggest specific improvements for configuration.
        
        Args:
            config: Configuration to analyze
            
        Returns:
            List of improvement suggestions
        """
        validation_result = self.validate_configuration(config, auto_correct=False)
        suggestions = []
        
        # Group issues by category
        memory_issues = [i for i in validation_result.issues if i.category == "memory"]
        cpu_issues = [i for i in validation_result.issues if i.category == "cpu"]
        training_issues = [i for i in validation_result.issues if i.category == "training"]
        
        # Generate specific suggestions
        if memory_issues:
            suggestions.append("Memory optimization: Consider reducing batch size or enabling dynamic batch sizing")
        
        if cpu_issues:
            suggestions.append("CPU optimization: Adjust worker count and thread settings for your hardware")
        
        if training_issues:
            suggestions.append("Training optimization: Review learning rate scaling and gradient accumulation")
        
        # Hardware-specific suggestions
        hardware_specs = self.get_hardware_specs()
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        if memory_gb < 8:
            suggestions.append("Low memory system: Enable memory monitoring and consider smaller model variants")
        
        if hardware_specs.cpu.cores < 4:
            suggestions.append("Limited CPU cores: Use conservative threading and smaller batch sizes")
        
        if not hardware_specs.platform.has_mkl:
            suggestions.append("No MKL detected: Consider installing Intel MKL for better performance")
        
        return suggestions
    
    def create_validation_report(self, config: Dict[str, Any]) -> str:
        """
        Create a comprehensive validation report.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Formatted validation report
        """
        validation_result = self.validate_configuration(config, auto_correct=True)
        hardware_specs = self.get_hardware_specs()
        
        report_lines = [
            "MacBook TRM Training Configuration Validation Report",
            "=" * 55,
            "",
            f"Hardware: {hardware_specs.cpu.brand}",
            f"Memory: {hardware_specs.memory.available_memory / (1024**3):.1f}GB available",
            f"CPU Cores: {hardware_specs.cpu.cores}",
            "",
            f"Compatibility Score: {validation_result.compatibility_score:.1f}/100",
            f"Overall Status: {'✓ Valid' if validation_result.is_valid else '✗ Invalid'}",
            "",
            "Issues Found:",
            "-" * 15
        ]
        
        if not validation_result.issues:
            report_lines.append("No issues detected - configuration is optimal!")
        else:
            for issue in validation_result.issues:
                level_symbol = {
                    ValidationLevel.CRITICAL: "🔴",
                    ValidationLevel.ERROR: "🟠", 
                    ValidationLevel.WARNING: "🟡",
                    ValidationLevel.INFO: "🔵"
                }.get(issue.level, "•")
                
                report_lines.append(f"{level_symbol} {issue.level.value.upper()}: {issue.message}")
                if issue.suggested_value is not None:
                    report_lines.append(f"   Suggested: {issue.suggested_value}")
                if issue.impact:
                    report_lines.append(f"   Impact: {issue.impact}")
                report_lines.append("")
        
        report_lines.extend([
            "",
            "Performance Assessment:",
            "-" * 22,
            validation_result.performance_impact,
            "",
            "Suggestions:",
            "-" * 12
        ])
        
        suggestions = self.suggest_configuration_improvements(config)
        for suggestion in suggestions:
            report_lines.append(f"• {suggestion}")
        
        if validation_result.corrected_config:
            report_lines.extend([
                "",
                "Auto-corrected configuration available.",
                "Use the corrected_config from ValidationResult for optimal settings."
            ])
        
        return "\n".join(report_lines)