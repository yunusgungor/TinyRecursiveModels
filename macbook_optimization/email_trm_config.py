"""
EmailTRM Configuration and Initialization for Production Training

This module provides configuration and initialization utilities for the real EmailTRM model
with recursive reasoning capabilities, optimized for MacBook hardware constraints.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig, create_email_trm_model
from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager


logger = logging.getLogger(__name__)


@dataclass
class ProductionEmailTRMConfig:
    """Production configuration for EmailTRM model with hardware optimization"""
    
    # Model architecture parameters
    vocab_size: int = 5000
    num_email_categories: int = 10
    hidden_size: int = 512
    L_layers: int = 2
    
    # Recursive reasoning parameters (optimized for email classification)
    H_cycles: int = 3  # Recursive reasoning cycles
    L_cycles: int = 4  # Layer cycles within each H cycle
    halt_max_steps: int = 8  # Maximum halting steps
    halt_exploration_prob: float = 0.1  # Exploration probability for halting
    
    # Email-specific features
    use_email_structure: bool = True  # Enable email structure awareness
    use_hierarchical_attention: bool = True  # Enable hierarchical attention
    use_category_embedding: bool = True  # Enable category embeddings
    use_email_pooling: bool = True  # Enable email-aware pooling
    
    # Email structure parameters
    email_structure_dim: int = 64  # Dimension for email structure embeddings
    subject_attention_weight: float = 2.0  # Weight for subject attention
    sender_attention_weight: float = 1.5  # Weight for sender attention
    pooling_strategy: str = "weighted"  # "mean", "max", "weighted", "attention"
    
    # Training optimization parameters
    classification_dropout: float = 0.1  # Dropout for classification head
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    memory_efficient_attention: bool = True  # Enable memory-efficient attention
    
    # Hardware-specific parameters (auto-configured)
    max_batch_size: int = 8  # Will be adjusted based on hardware
    max_sequence_length: int = 512  # Maximum email sequence length
    optimal_dtype: str = "float32"  # Optimal data type for hardware
    num_workers: int = 2  # Number of data loading workers
    
    # Email token IDs (should match dataset preprocessing)
    pad_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    subject_id: int = 3
    body_id: int = 4
    from_id: int = 5
    to_id: int = 6
    
    # Production targets
    target_accuracy: float = 0.95
    min_category_accuracy: float = 0.90


class EmailTRMConfigManager:
    """Manager for EmailTRM configuration and initialization"""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.memory_manager = MemoryManager()
        self._hardware_summary = None
        
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get hardware summary for configuration optimization"""
        if self._hardware_summary is None:
            self._hardware_summary = self.hardware_detector.get_hardware_summary()
        return self._hardware_summary
    
    def create_optimized_config(self, 
                              vocab_size: int,
                              base_config: Optional[ProductionEmailTRMConfig] = None) -> ProductionEmailTRMConfig:
        """
        Create optimized EmailTRM configuration based on hardware capabilities
        
        Args:
            vocab_size: Vocabulary size from dataset
            base_config: Base configuration to modify (optional)
            
        Returns:
            Optimized ProductionEmailTRMConfig
        """
        
        # Start with base config or default
        config = base_config or ProductionEmailTRMConfig()
        config.vocab_size = vocab_size
        
        # Get hardware specifications
        hw_summary = self.get_hardware_summary()
        
        logger.info(f"Optimizing EmailTRM config for hardware: {hw_summary['cpu']['cores']} cores, "
                   f"{hw_summary['memory']['total_gb']:.1f}GB RAM")
        
        # Optimize based on available memory
        total_memory_gb = hw_summary['memory']['total_gb']
        available_memory_gb = hw_summary['memory']['available_gb']
        
        # Configure model size based on memory constraints
        if available_memory_gb >= 12:
            # High memory configuration
            config.hidden_size = 768
            config.L_layers = 3
            config.H_cycles = 4
            config.L_cycles = 5
            config.max_batch_size = 16
            config.email_structure_dim = 128
            logger.info("Using high-memory configuration")
            
        elif available_memory_gb >= 8:
            # Medium memory configuration
            config.hidden_size = 512
            config.L_layers = 2
            config.H_cycles = 3
            config.L_cycles = 4
            config.max_batch_size = 8
            config.email_structure_dim = 64
            logger.info("Using medium-memory configuration")
            
        else:
            # Low memory configuration
            config.hidden_size = 256
            config.L_layers = 2
            config.H_cycles = 2
            config.L_cycles = 3
            config.max_batch_size = 4
            config.email_structure_dim = 32
            config.gradient_checkpointing = True
            logger.info("Using low-memory configuration")
        
        # Configure based on CPU capabilities
        cpu_cores = hw_summary['cpu']['cores']
        config.num_workers = min(cpu_cores, 4)  # Conservative worker count
        
        # Configure data type based on platform capabilities
        config.optimal_dtype = hw_summary['platform']['optimal_dtype']
        
        # Enable optimizations based on CPU features
        if hw_summary['platform']['supports_avx2']:
            # AVX2 support enables better vectorization
            config.memory_efficient_attention = True
            logger.info("Enabled AVX2 optimizations")
        
        # Configure halting parameters for email classification
        # Emails typically need fewer reasoning cycles than complex puzzles
        config.halt_max_steps = min(8, config.H_cycles * 2)
        config.halt_exploration_prob = 0.1  # Conservative exploration
        
        # Adjust sequence length based on memory constraints
        if available_memory_gb < 6:
            config.max_sequence_length = 256  # Shorter sequences for memory-constrained systems
        
        logger.info(f"Optimized config: {config.hidden_size}d hidden, "
                   f"{config.H_cycles}H/{config.L_cycles}L cycles, "
                   f"batch_size={config.max_batch_size}")
        
        return config
    
    def create_email_trm_config(self, production_config: ProductionEmailTRMConfig) -> EmailTRMConfig:
        """
        Convert ProductionEmailTRMConfig to EmailTRMConfig for model initialization
        
        Args:
            production_config: Production configuration
            
        Returns:
            EmailTRMConfig for model initialization
        """
        
        return EmailTRMConfig(
            # Basic model parameters
            vocab_size=production_config.vocab_size,
            hidden_size=production_config.hidden_size,
            L_layers=production_config.L_layers,
            
            # Recursive reasoning parameters
            H_cycles=production_config.H_cycles,
            L_cycles=production_config.L_cycles,
            halt_max_steps=production_config.halt_max_steps,
            halt_exploration_prob=production_config.halt_exploration_prob,
            
            # Email-specific parameters
            num_email_categories=production_config.num_email_categories,
            use_category_embedding=production_config.use_category_embedding,
            classification_dropout=production_config.classification_dropout,
            
            # Email structure awareness
            use_email_structure=production_config.use_email_structure,
            email_structure_dim=production_config.email_structure_dim,
            
            # Attention mechanisms
            use_hierarchical_attention=production_config.use_hierarchical_attention,
            subject_attention_weight=production_config.subject_attention_weight,
            sender_attention_weight=production_config.sender_attention_weight,
            
            # Pooling strategy
            use_email_pooling=production_config.use_email_pooling,
            pooling_strategy=production_config.pooling_strategy,
            
            # Token IDs
            pad_id=production_config.pad_id,
            eos_id=production_config.eos_id,
            unk_id=production_config.unk_id,
            subject_id=production_config.subject_id,
            body_id=production_config.body_id,
            from_id=production_config.from_id,
            to_id=production_config.to_id,
        )
    
    def initialize_email_trm_model(self, 
                                  production_config: ProductionEmailTRMConfig,
                                  device: Optional[torch.device] = None) -> EmailTRM:
        """
        Initialize EmailTRM model with optimized configuration
        
        Args:
            production_config: Production configuration
            device: Target device (auto-detected if None)
            
        Returns:
            Initialized EmailTRM model
        """
        
        if device is None:
            device = torch.device("cpu")  # MacBook training uses CPU
        
        # Convert to EmailTRMConfig
        email_config = self.create_email_trm_config(production_config)
        
        logger.info(f"Initializing EmailTRM model with config:")
        logger.info(f"  - Vocab size: {email_config.vocab_size}")
        logger.info(f"  - Hidden size: {email_config.hidden_size}")
        logger.info(f"  - Layers: {email_config.L_layers}")
        logger.info(f"  - H cycles: {email_config.H_cycles}, L cycles: {email_config.L_cycles}")
        logger.info(f"  - Email categories: {email_config.num_email_categories}")
        logger.info(f"  - Email structure awareness: {email_config.use_email_structure}")
        logger.info(f"  - Hierarchical attention: {email_config.use_hierarchical_attention}")
        
        # Initialize model
        model = EmailTRM(email_config)
        model = model.to(device)
        
        # Enable gradient checkpointing if configured
        if production_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        
        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized successfully:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Model size: ~{total_params * 4 / 1024**2:.1f}MB (float32)")
        
        return model
    
    def validate_model_configuration(self, model: EmailTRM, production_config: ProductionEmailTRMConfig) -> Dict[str, Any]:
        """
        Validate model configuration and estimate resource requirements
        
        Args:
            model: Initialized EmailTRM model
            production_config: Production configuration
            
        Returns:
            Validation results and resource estimates
        """
        
        validation_results = {
            "model_valid": True,
            "warnings": [],
            "resource_estimates": {},
            "configuration_summary": {}
        }
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / 1024**2  # Assuming float32
        
        # Estimate memory requirements
        hw_summary = self.get_hardware_summary()
        available_memory_gb = hw_summary['memory']['available_gb']
        
        # Rough memory estimation for training
        # Model weights + gradients + optimizer states + activations
        estimated_memory_gb = (
            model_size_mb * 3 / 1024 +  # Model + gradients + optimizer
            production_config.max_batch_size * production_config.max_sequence_length * 
            production_config.hidden_size * 4 / 1024**3  # Activations
        )
        
        validation_results["resource_estimates"] = {
            "model_size_mb": model_size_mb,
            "estimated_memory_gb": estimated_memory_gb,
            "available_memory_gb": available_memory_gb,
            "memory_utilization": estimated_memory_gb / available_memory_gb
        }
        
        # Check for potential issues
        if estimated_memory_gb > available_memory_gb * 0.8:
            validation_results["warnings"].append(
                f"High memory usage estimated ({estimated_memory_gb:.1f}GB > 80% of available)"
            )
        
        if production_config.max_batch_size > 16:
            validation_results["warnings"].append(
                f"Large batch size ({production_config.max_batch_size}) may cause memory issues"
            )
        
        # Configuration summary
        validation_results["configuration_summary"] = {
            "recursive_reasoning_enabled": production_config.H_cycles > 1,
            "email_structure_awareness": production_config.use_email_structure,
            "hierarchical_attention": production_config.use_hierarchical_attention,
            "category_embeddings": production_config.use_category_embedding,
            "gradient_checkpointing": production_config.gradient_checkpointing,
            "memory_efficient_attention": production_config.memory_efficient_attention
        }
        
        return validation_results
    
    def save_config(self, config: ProductionEmailTRMConfig, filepath: Path) -> None:
        """Save configuration to file"""
        config_dict = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: Path) -> ProductionEmailTRMConfig:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        config = ProductionEmailTRMConfig(**config_dict)
        logger.info(f"Configuration loaded from {filepath}")
        return config


def create_production_email_trm(vocab_size: int, 
                               base_config: Optional[ProductionEmailTRMConfig] = None,
                               device: Optional[torch.device] = None) -> Tuple[EmailTRM, ProductionEmailTRMConfig]:
    """
    Convenience function to create production-ready EmailTRM model
    
    Args:
        vocab_size: Vocabulary size from dataset
        base_config: Base configuration to modify (optional)
        device: Target device (auto-detected if None)
        
    Returns:
        Tuple of (initialized model, final configuration)
    """
    
    config_manager = EmailTRMConfigManager()
    
    # Create optimized configuration
    production_config = config_manager.create_optimized_config(vocab_size, base_config)
    
    # Initialize model
    model = config_manager.initialize_email_trm_model(production_config, device)
    
    # Validate configuration
    validation_results = config_manager.validate_model_configuration(model, production_config)
    
    if validation_results["warnings"]:
        logger.warning("Configuration warnings:")
        for warning in validation_results["warnings"]:
            logger.warning(f"  - {warning}")
    
    logger.info("Production EmailTRM model created successfully")
    
    return model, production_config


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    vocab_size = 5000
    model, config = create_production_email_trm(vocab_size)
    
    print(f"Model created with configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - H cycles: {config.H_cycles}, L cycles: {config.L_cycles}")
    print(f"  - Max batch size: {config.max_batch_size}")
    print(f"  - Email structure awareness: {config.use_email_structure}")
    print(f"  - Hierarchical attention: {config.use_hierarchical_attention}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.num_email_categories, (batch_size,))
    
    with torch.no_grad():
        outputs = model(inputs, labels=labels)
    
    print(f"\nTest forward pass successful:")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Output logits shape: {outputs['logits'].shape}")
    print(f"  - Number of reasoning cycles: {outputs['num_cycles']}")
    print(f"  - Loss: {outputs['loss'].item():.4f}")