"""
Email Training Configuration System for MacBook Optimization

This module provides email-specific training configuration and adaptation
for MacBook hardware constraints, extending the base TrainingConfigAdapter
with email classification specific parameters and optimizations.
"""

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .training_config_adapter import TrainingConfigAdapter, HardwareSpecs, TrainingParams, ConfigurationResult
from .hardware_detection import HardwareDetector
from .memory_management import MemoryManager
from .cpu_optimization import CPUOptimizer
from .config_management import MacBookTrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class EmailTrainingConfig:
    """Email-specific training configuration with MacBook optimizations."""
    
    # Model parameters
    model_name: str = "EmailTRM"
    vocab_size: int = 5000
    hidden_size: int = 512
    num_layers: int = 2
    num_email_categories: int = 10
    
    # Training parameters
    batch_size: int = 8  # Adjusted for MacBook
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    max_steps: int = 10000
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "exponential"
    warmup_steps: int = 100
    
    # Email-specific parameters
    max_sequence_length: int = 512
    use_email_structure: bool = True
    subject_attention_weight: float = 2.0
    pooling_strategy: str = "weighted"
    
    # Email tokenization parameters
    min_token_frequency: int = 2
    special_token_ratio: float = 0.1  # Ratio of special tokens in vocab
    domain_feature_weight: float = 1.5
    content_feature_weight: float = 1.2
    
    # Email dataset parameters
    email_augmentation_prob: float = 0.3
    category_balancing: bool = True
    cross_validation_folds: int = 5
    
    # MacBook optimization parameters
    memory_limit_mb: int = 6000
    enable_memory_monitoring: bool = True
    dynamic_batch_sizing: bool = True
    use_cpu_optimization: bool = True
    num_workers: int = 2
    
    # Performance targets
    target_accuracy: float = 0.95
    min_category_accuracy: float = 0.90
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Loss function weights
    classification_loss_weight: float = 1.0
    halt_loss_weight: float = 0.01
    contrastive_loss_weight: float = 0.1
    calibration_loss_weight: float = 0.05
    
    # Training control
    max_grad_norm: float = 1.0
    log_interval: int = 50
    eval_interval: int = 200
    
    # Email-specific optimization flags
    enable_subject_prioritization: bool = True
    enable_sender_analysis: bool = True
    enable_content_features: bool = True
    use_hierarchical_attention: bool = True


@dataclass
class EmailTrainingParams(TrainingParams):
    """Extended training parameters for email classification."""
    
    # Email-specific parameters
    email_vocab_size: int = 5000
    num_email_categories: int = 10
    subject_attention_weight: float = 2.0
    
    # Email preprocessing parameters
    email_augmentation_enabled: bool = True
    category_balancing_enabled: bool = True
    
    # Email model parameters
    use_hierarchical_attention: bool = True
    enable_subject_prioritization: bool = True
    
    # Email dataset parameters
    email_streaming_threshold_mb: float = 300.0
    email_cache_threshold_mb: float = 150.0


class EmailTrainingConfigAdapter(TrainingConfigAdapter):
    """Adapter for email classification training configurations on MacBook hardware."""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        """
        Initialize email training configuration adapter.
        
        Args:
            hardware_detector: Hardware detector instance (created if None)
        """
        super().__init__(hardware_detector)
        
        # Email-specific optimization parameters
        self.email_model_params = 7_000_000  # EmailTRM parameters
        self.email_categories = 10
        self.default_email_vocab_size = 5000
        
    def adapt_email_config(self, base_config: EmailTrainingConfig, 
                          dataset_size: int,
                          hardware_specs: Optional[HardwareSpecs] = None) -> Dict[str, Any]:
        """
        Adapt email training configuration for MacBook hardware constraints.
        
        Args:
            base_config: Base email training configuration
            dataset_size: Size of email training dataset
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Adapted configuration dictionary
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
        
        # Convert dataclass to dict for processing
        config_dict = asdict(base_config)
        
        # Calculate email-specific memory requirements
        email_memory_req = self._calculate_email_memory_requirements(
            base_config, dataset_size, hardware_specs
        )
        
        # Adapt batch size for email sequences
        adapted_batch_size = self._calculate_email_batch_size(
            base_config, email_memory_req, hardware_specs
        )
        
        config_dict['batch_size'] = adapted_batch_size
        
        # Calculate gradient accumulation for effective batch size
        target_effective_batch = max(32, dataset_size // 1000)  # Adaptive target
        gradient_accumulation = max(1, target_effective_batch // adapted_batch_size)
        config_dict['gradient_accumulation_steps'] = gradient_accumulation
        
        # Adjust learning rate for email classification
        effective_batch_size = adapted_batch_size * gradient_accumulation
        lr_scale = math.sqrt(effective_batch_size / 32)  # Scale from base of 32
        config_dict['learning_rate'] = base_config.learning_rate * lr_scale
        
        # Adapt sequence length based on memory constraints
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        if memory_gb < 8:
            config_dict['max_sequence_length'] = min(base_config.max_sequence_length, 256)
        elif memory_gb < 12:
            config_dict['max_sequence_length'] = min(base_config.max_sequence_length, 384)
        
        # Adapt vocabulary size based on memory and dataset size
        vocab_scale = min(1.0, dataset_size / 10000)  # Scale vocab with dataset size
        memory_scale = min(1.0, memory_gb / 8)  # Scale with available memory
        adapted_vocab_size = int(base_config.vocab_size * vocab_scale * memory_scale)
        config_dict['vocab_size'] = max(1000, adapted_vocab_size)  # Minimum 1000 tokens
        
        # Adapt model complexity for memory constraints
        if memory_gb < 6:
            config_dict['hidden_size'] = min(base_config.hidden_size, 256)
            config_dict['num_layers'] = min(base_config.num_layers, 1)
        elif memory_gb < 10:
            config_dict['hidden_size'] = min(base_config.hidden_size, 384)
        
        # Set email-specific optimization flags
        config_dict['enable_subject_prioritization'] = memory_gb >= 4  # Disable on very low memory
        config_dict['use_hierarchical_attention'] = memory_gb >= 6
        config_dict['enable_content_features'] = memory_gb >= 6  # Enable with 6GB+ memory
        
        # MacBook-specific optimizations
        config_dict['num_workers'] = min(hardware_specs.optimal_workers, 4)  # Limit for email processing
        config_dict['memory_limit_mb'] = int(hardware_specs.memory.available_memory / (1024**2) * 0.7)
        
        # Email dataset streaming thresholds
        config_dict['email_streaming_threshold_mb'] = memory_gb * 30  # 30MB per GB of RAM
        config_dict['email_cache_threshold_mb'] = memory_gb * 15  # 15MB per GB of RAM
        
        return config_dict
    
    def _calculate_email_memory_requirements(self, config: EmailTrainingConfig,
                                           dataset_size: int,
                                           hardware_specs: HardwareSpecs) -> Dict[str, float]:
        """Calculate memory requirements for email training."""
        
        # Base model memory (EmailTRM with embeddings)
        model_memory_mb = (self.email_model_params * 4) / (1024**2)  # 4 bytes per param
        
        # Vocabulary memory (embeddings)
        vocab_memory_mb = (config.vocab_size * config.hidden_size * 4) / (1024**2)
        
        # Sequence memory per batch
        seq_memory_per_sample = config.max_sequence_length * config.hidden_size * 4 / (1024**2)
        batch_memory_mb = seq_memory_per_sample * config.batch_size
        
        # Email-specific feature memory
        email_features_mb = 0
        if config.enable_subject_prioritization:
            email_features_mb += batch_memory_mb * 0.1  # 10% overhead for subject attention
        if config.use_hierarchical_attention:
            email_features_mb += batch_memory_mb * 0.15  # 15% overhead for hierarchical attention
        if config.enable_content_features:
            email_features_mb += batch_memory_mb * 0.05  # 5% overhead for content features
        
        # Gradient memory (same as model)
        gradient_memory_mb = model_memory_mb
        
        # Optimizer memory (Adam: 2x model params)
        optimizer_memory_mb = model_memory_mb * 2
        
        # Total memory requirement
        total_memory_mb = (model_memory_mb + vocab_memory_mb + batch_memory_mb + 
                          email_features_mb + gradient_memory_mb + optimizer_memory_mb)
        
        return {
            'model_memory_mb': model_memory_mb,
            'vocab_memory_mb': vocab_memory_mb,
            'batch_memory_mb': batch_memory_mb,
            'email_features_mb': email_features_mb,
            'gradient_memory_mb': gradient_memory_mb,
            'optimizer_memory_mb': optimizer_memory_mb,
            'total_memory_mb': total_memory_mb
        }
    
    def _calculate_email_batch_size(self, config: EmailTrainingConfig,
                                   memory_req: Dict[str, float],
                                   hardware_specs: HardwareSpecs) -> int:
        """Calculate optimal batch size for email training."""
        
        available_memory_mb = hardware_specs.memory.available_memory / (1024**2) * 0.8  # 80% utilization
        
        # Fixed memory (model, vocab, gradients, optimizer)
        fixed_memory_mb = (memory_req['model_memory_mb'] + memory_req['vocab_memory_mb'] + 
                          memory_req['gradient_memory_mb'] + memory_req['optimizer_memory_mb'])
        
        # Available memory for batches
        batch_memory_available = available_memory_mb - fixed_memory_mb
        
        if batch_memory_available <= 0:
            logger.warning("Insufficient memory for training - using minimum batch size")
            return 1
        
        # Memory per sample (including email features)
        memory_per_sample = (config.max_sequence_length * config.hidden_size * 4) / (1024**2)
        
        # Add email feature overhead
        if config.enable_subject_prioritization:
            memory_per_sample *= 1.1
        if config.use_hierarchical_attention:
            memory_per_sample *= 1.15
        if config.enable_content_features:
            memory_per_sample *= 1.05
        
        # Calculate maximum batch size
        max_batch_size = int(batch_memory_available / memory_per_sample)
        
        # Constrain to reasonable limits for email classification
        optimal_batch_size = min(max_batch_size, 16)  # Max 16 for email training
        optimal_batch_size = max(optimal_batch_size, 1)  # Min 1
        
        # Prefer power of 2 for efficiency
        if optimal_batch_size >= 8:
            optimal_batch_size = 8
        elif optimal_batch_size >= 4:
            optimal_batch_size = 4
        elif optimal_batch_size >= 2:
            optimal_batch_size = 2
        
        return optimal_batch_size
    
    def calculate_email_training_parameters(self, config: EmailTrainingConfig,
                                          dataset_size: int,
                                          hardware_specs: Optional[HardwareSpecs] = None) -> EmailTrainingParams:
        """
        Calculate email-specific training parameters.
        
        Args:
            config: Email training configuration
            dataset_size: Size of email dataset
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Email training parameters
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
        
        # Get base training parameters
        base_params = self.calculate_training_parameters(dataset_size, hardware_specs)
        
        # Calculate email-specific memory requirements
        memory_req = self._calculate_email_memory_requirements(config, dataset_size, hardware_specs)
        
        # Calculate email-specific batch size
        email_batch_size = self._calculate_email_batch_size(config, memory_req, hardware_specs)
        
        # Calculate gradient accumulation for email training
        target_effective_batch = max(32, dataset_size // 500)  # Larger effective batch for email
        gradient_accumulation = max(1, target_effective_batch // email_batch_size)
        
        # Email-specific learning rate scaling
        effective_batch_size = email_batch_size * gradient_accumulation
        lr_scale = math.sqrt(effective_batch_size / 32)
        email_learning_rate = config.learning_rate * lr_scale
        
        # Memory thresholds for email datasets
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        email_streaming_threshold = memory_gb * 30  # 30MB per GB
        email_cache_threshold = memory_gb * 15  # 15MB per GB
        
        return EmailTrainingParams(
            # Base parameters
            batch_size=email_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            effective_batch_size=effective_batch_size,
            learning_rate=email_learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=min(100, dataset_size // effective_batch_size // 10),
            
            # Hardware parameters
            num_workers=min(hardware_specs.optimal_workers, 4),
            pin_memory=False,
            torch_threads=min(hardware_specs.cpu.cores, 4),
            
            # Memory management
            memory_limit_mb=int(memory_req['total_memory_mb'] * 1.2),  # 20% buffer
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            
            # Model parameters
            max_sequence_length=config.max_sequence_length,
            model_complexity_factor=1.0 if memory_gb >= 8 else min(1.0, memory_gb / 8),
            
            # Optimization flags
            use_mkl=hardware_specs.platform.has_mkl,
            enable_cpu_optimization=True,
            enable_mixed_precision=False,
            
            # Checkpointing
            checkpoint_interval=500 if memory_gb < 12 else 1000,
            max_checkpoints_to_keep=3,
            
            # Email-specific parameters
            email_vocab_size=config.vocab_size,
            num_email_categories=config.num_email_categories,
            subject_attention_weight=config.subject_attention_weight,
            
            # Email preprocessing
            email_augmentation_enabled=config.email_augmentation_prob > 0,
            category_balancing_enabled=config.category_balancing,
            
            # Email model features
            use_hierarchical_attention=config.use_hierarchical_attention and memory_gb >= 6,
            enable_subject_prioritization=config.enable_subject_prioritization and memory_gb >= 4,
            
            # Email dataset management
            email_streaming_threshold_mb=email_streaming_threshold,
            email_cache_threshold_mb=email_cache_threshold
        )
    
    def create_email_hardware_config(self, base_config: EmailTrainingConfig,
                                   dataset_size: int) -> ConfigurationResult:
        """
        Create complete email training configuration for current hardware.
        
        Args:
            base_config: Base email training configuration
            dataset_size: Size of email dataset
            
        Returns:
            Complete configuration result with email-specific adaptations
        """
        hardware_specs = self.get_hardware_specs()
        
        # Adapt email configuration
        adapted_config = self.adapt_email_config(base_config, dataset_size, hardware_specs)
        
        # Calculate email training parameters
        email_params = self.calculate_email_training_parameters(base_config, dataset_size, hardware_specs)
        
        # Generate validation warnings
        warnings = []
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        if memory_gb < 6:
            warnings.append(f"Limited memory ({memory_gb:.1f}GB) - email features may be disabled")
        
        if email_params.batch_size < 2:
            warnings.append(f"Very small batch size ({email_params.batch_size}) may impact email classification accuracy")
        
        if not email_params.use_hierarchical_attention:
            warnings.append("Hierarchical attention disabled due to memory constraints")
        
        if adapted_config['vocab_size'] < base_config.vocab_size:
            warnings.append(f"Vocabulary size reduced from {base_config.vocab_size} to {adapted_config['vocab_size']}")
        
        # Performance estimates for email training
        estimated_samples_per_second = self._estimate_email_training_speed(email_params, hardware_specs)
        estimated_epoch_time = dataset_size / estimated_samples_per_second if estimated_samples_per_second > 0 else 0
        
        performance_estimates = {
            'estimated_samples_per_second': estimated_samples_per_second,
            'estimated_epoch_time_minutes': estimated_epoch_time / 60,
            'memory_utilization_percent': (email_params.memory_limit_mb / 
                                         (hardware_specs.memory.available_memory / (1024**2))) * 100,
            'cpu_utilization_percent': min(100, email_params.torch_threads / hardware_specs.cpu.cores * 100),
            'estimated_accuracy_target_epochs': self._estimate_epochs_to_target(dataset_size, email_params)
        }
        
        # Generate reasoning
        reasoning = self._generate_email_configuration_reasoning(
            base_config, email_params, hardware_specs, warnings
        )
        
        return ConfigurationResult(
            adapted_config=adapted_config,
            training_params=email_params,
            hardware_specs=hardware_specs,
            validation_warnings=warnings,
            performance_estimates=performance_estimates,
            reasoning=reasoning
        )
    
    def _estimate_email_training_speed(self, params: EmailTrainingParams,
                                     hardware_specs: HardwareSpecs) -> float:
        """Estimate email training speed in samples per second."""
        
        # Base speed for email classification (slower than general TRM due to email features)
        base_speed = 8.0  # samples/second baseline for email
        
        # Hardware adjustments
        cpu_factor = min(2.0, hardware_specs.cpu.base_frequency / 2.4)
        core_factor = min(1.4, hardware_specs.cpu.cores / 4)
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        memory_factor = min(1.2, memory_gb / 8)
        
        # Batch size efficiency
        batch_factor = min(1.2, params.batch_size / 4)
        
        # Email-specific overhead
        email_overhead = 1.0
        if params.use_hierarchical_attention:
            email_overhead *= 0.9  # 10% slower
        if params.enable_subject_prioritization:
            email_overhead *= 0.95  # 5% slower
        
        # Optimization factors
        opt_factor = 1.0
        if params.use_mkl:
            opt_factor *= 1.15
        if hardware_specs.platform.supports_avx2:
            opt_factor *= 1.1
        
        estimated_speed = (base_speed * cpu_factor * core_factor * memory_factor * 
                          batch_factor * email_overhead * opt_factor)
        
        return max(0.1, estimated_speed)
    
    def _estimate_epochs_to_target(self, dataset_size: int, params: EmailTrainingParams) -> int:
        """Estimate epochs needed to reach 95% accuracy target."""
        
        # Base estimate: larger datasets need fewer epochs
        if dataset_size > 50000:
            base_epochs = 3
        elif dataset_size > 10000:
            base_epochs = 5
        else:
            base_epochs = 8
        
        # Adjust for batch size (smaller batches need more epochs)
        batch_adjustment = max(1.0, 8 / params.effective_batch_size)
        
        # Adjust for email features (better features converge faster)
        feature_factor = 1.0
        if params.use_hierarchical_attention:
            feature_factor *= 0.9
        if params.enable_subject_prioritization:
            feature_factor *= 0.95
        
        estimated_epochs = int(base_epochs * batch_adjustment * feature_factor)
        return max(2, min(15, estimated_epochs))  # Between 2-15 epochs
    
    def _generate_email_configuration_reasoning(self, config: EmailTrainingConfig,
                                              params: EmailTrainingParams,
                                              hardware_specs: HardwareSpecs,
                                              warnings: List[str]) -> str:
        """Generate reasoning for email configuration choices."""
        
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        reasoning_parts = [
            f"Email classification configuration adapted for {hardware_specs.cpu.brand} with {hardware_specs.cpu.cores} cores and {memory_gb:.1f}GB memory.",
            f"Batch size: {params.batch_size}, gradient accumulation: {params.gradient_accumulation_steps}, effective batch: {params.effective_batch_size}.",
            f"Email vocabulary: {params.email_vocab_size} tokens, sequence length: {params.max_sequence_length}.",
        ]
        
        if params.use_hierarchical_attention:
            reasoning_parts.append("Hierarchical attention enabled for email structure awareness.")
        
        if params.enable_subject_prioritization:
            reasoning_parts.append("Subject prioritization enabled for better classification.")
        
        if params.email_streaming_threshold_mb < 500:
            reasoning_parts.append("Streaming enabled for large email datasets due to memory constraints.")
        
        if warnings:
            reasoning_parts.append(f"Generated {len(warnings)} optimization warnings.")
        
        return " ".join(reasoning_parts)
    
    def get_email_configuration_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get email-specific configuration templates for different MacBook setups."""
        
        templates = {}
        
        # 8GB MacBook template
        templates['email_macbook_8gb'] = {
            'description': 'Email classification optimized for 8GB MacBook',
            'config': EmailTrainingConfig(
                batch_size=2,
                gradient_accumulation_steps=16,
                vocab_size=3000,
                hidden_size=256,
                num_layers=2,
                max_sequence_length=256,
                use_hierarchical_attention=False,
                enable_subject_prioritization=True,
                enable_content_features=False
            ),
            'recommended_for': ['MacBook Air 8GB', 'MacBook Pro 13" 8GB']
        }
        
        # 16GB MacBook template
        templates['email_macbook_16gb'] = {
            'description': 'Email classification optimized for 16GB MacBook',
            'config': EmailTrainingConfig(
                batch_size=4,
                gradient_accumulation_steps=8,
                vocab_size=5000,
                hidden_size=512,
                num_layers=2,
                max_sequence_length=512,
                use_hierarchical_attention=True,
                enable_subject_prioritization=True,
                enable_content_features=True
            ),
            'recommended_for': ['MacBook Pro 13" 16GB', 'MacBook Pro 16" base']
        }
        
        # High-end MacBook template
        templates['email_macbook_32gb'] = {
            'description': 'Email classification optimized for 32GB+ MacBook',
            'config': EmailTrainingConfig(
                batch_size=8,
                gradient_accumulation_steps=4,
                vocab_size=8000,
                hidden_size=512,
                num_layers=3,
                max_sequence_length=512,
                use_hierarchical_attention=True,
                enable_subject_prioritization=True,
                enable_content_features=True,
                subject_attention_weight=3.0
            ),
            'recommended_for': ['MacBook Pro 16" high-end', 'Mac Studio']
        }
        
        return templates
    
    def recommend_email_template(self, dataset_size: int,
                               hardware_specs: Optional[HardwareSpecs] = None) -> str:
        """Recommend email configuration template based on hardware and dataset."""
        
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
        
        memory_gb = hardware_specs.memory.total_memory / (1024**3)
        cpu_cores = hardware_specs.cpu.cores
        
        # Consider dataset size in recommendation
        if dataset_size > 100000:  # Large dataset
            if memory_gb >= 24 and cpu_cores >= 6:
                return 'email_macbook_32gb'
            elif memory_gb >= 14:
                return 'email_macbook_16gb'
            else:
                return 'email_macbook_8gb'
        else:  # Smaller dataset
            if memory_gb >= 14 and cpu_cores >= 4:
                return 'email_macbook_16gb'
            elif memory_gb >= 7:
                return 'email_macbook_8gb'
            else:
                return 'email_macbook_8gb'  # Conservative fallback